#include <cstdint>
#include <glm/detail/qualifier.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdio.h>
#include <iostream>
#include <random>
#include <limits>

#include "include/cpu_helpers.hpp"

#include <curand_kernel.h>
#include "cuda/helpers.cuh"
#include "cuda_check.cuh"
#include <cub/cub.cuh>

#include "cuda/gaussian.cuh"
#include "include/defines.hpp"
#include "include/ppm.hpp"
#include "include/renderer.hpp"
#include "renderer.hpp"
#include "gaussian.cuh"

#include "nvtx3/nvtx3.hpp"

namespace
{
    constexpr uint32_t MAX = 255;

    constexpr uint32_t WIDTH = 800;
    constexpr uint32_t HEIGHT = 600;

    constexpr uint32_t GAUSSIANS_COUNT = 10000;

    constexpr float ASPECT_RATIO = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);
    constexpr float NEAR = 0.1f;
    constexpr float FAR = 100.0f;
    constexpr float FOV = 90.0f;

    const glm::vec3 cameraPos = glm::vec3(0.0f ,0.0f, 2.0f);
    const glm::vec3 targetPos = glm::vec3(0.0f ,0.0f, 0.0f);
    const glm::vec3 GLOBAL_UP = glm::vec3(0.0f, 1.0f, 0.0f);
}

using D_GS::pixel;
using D_GS::Gaussian;

int main()
{
    H_GS::Renderer renderer;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist_x(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_z(-30.0f, -5.0f);
    std::uniform_real_distribution<float> dist_color(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist_opacity(0.0f, 1.0f);

    std::vector<Gaussian> h_gaussians;
    h_gaussians.reserve(GAUSSIANS_COUNT);

    for(uint32_t i = 0; i < GAUSSIANS_COUNT; i++)
    {
        h_gaussians.emplace_back(Gaussian{
                .rotation = make_float4(1.0f, 0.0f, 0.0f, 0.0f),
                .mean3d = make_float3(dist_x(gen), dist_y(gen), dist_z(gen)),
                .scale = make_float3(1.0f, 1.0f, 1.0f),
                .color = make_float3(dist_color(gen), dist_opacity(gen), dist_color(gen)),
                .opacity = dist_opacity(gen)
            }
        );
    }

    size_t gc = h_gaussians.size();

    glm::mat4 projMat = glm::perspective(
        glm::radians(FOV),
        ASPECT_RATIO,
        NEAR,
        FAR
    );

    const float FOCAL_X = projMat[0][0] * WIDTH * 0.5f;
    const float FOCAL_Y = projMat[1][1] * HEIGHT * 0.5f;

    glm::mat4 viewMat = glm::lookAt(cameraPos, targetPos, GLOBAL_UP);

    std::vector<H_GS::pixel>& img = renderer.getImage();

    pixel* d_image = nullptr;

    Gaussian* d_gaussians = nullptr;
    float* d_cov3Ds = nullptr;
    float3* d_cov2Ds = nullptr;
    float3* d_inv_cov = nullptr;
    float2* d_pix_mean = nullptr;
    float* d_depth = nullptr;
    float* d_proj_mat = nullptr;
    float* d_view_mat = nullptr;
    uint32_t* d_tiles_touched = nullptr;
    uint32_t* d_tiles_offset = nullptr;
    uint64_t* d_keys_unsorted = nullptr;
    uint64_t* d_keys_sorted = nullptr;
    uint32_t* d_values_unsorted = nullptr;
    uint32_t* d_values_sorted = nullptr;
    uint2* d_tiles_range = nullptr;

    size_t ele_num = img.size();

    uint32_t image_size = static_cast<uint32_t>(ele_num);
    size_t image_bytes = ele_num * sizeof(pixel);


    size_t gaussian_size = gc * sizeof(D_GS::Gaussian);
    size_t gs_float_size = gc * sizeof(float);
    size_t gs_float2_size = gc * sizeof(float2);
    size_t gs_float3_size = gc * sizeof(float3);
    size_t gs_uint32_size = gc * sizeof(uint32_t);
    size_t inv_cov_size = gc * sizeof(float3);
    size_t mat4x4_size_f = 16 * sizeof(float);

    nvtx3::range_handle h = nvtx3::start_range("First CudaMalloc batch", nvtx3::rgb{127,255,0});

    CUDA_CHECK(cudaMalloc((void**)&d_image, image_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_gaussians, gaussian_size));

    CUDA_CHECK(cudaMalloc((void**)&d_cov3Ds, gs_float_size * 6));
    CUDA_CHECK(cudaMalloc((void**)&d_cov2Ds, gs_float3_size));
    CUDA_CHECK(cudaMalloc((void**)&d_inv_cov, inv_cov_size));
    CUDA_CHECK(cudaMalloc((void**)&d_pix_mean, gs_float2_size));
    CUDA_CHECK(cudaMalloc((void**)&d_depth, gs_float_size));
    CUDA_CHECK(cudaMalloc((void**)&d_proj_mat, mat4x4_size_f));
    CUDA_CHECK(cudaMalloc((void**)&d_view_mat, mat4x4_size_f));
    CUDA_CHECK(cudaMalloc((void**)&d_tiles_touched, gs_uint32_size));
    CUDA_CHECK(cudaMalloc((void**)&d_tiles_offset, gs_uint32_size));

    nvtx3::end_range(h);

    CUDA_CHECK(cudaMemcpy(d_image, img.data(), image_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_gaussians, h_gaussians.data(), gaussian_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_proj_mat, glm::value_ptr(projMat), mat4x4_size_f, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_view_mat, glm::value_ptr(viewMat), mat4x4_size_f, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_cov3Ds, 0, gs_float_size));
    CUDA_CHECK(cudaMemset(d_cov2Ds, 0, gs_float3_size));
    CUDA_CHECK(cudaMemset(d_inv_cov, 0, inv_cov_size));
    CUDA_CHECK(cudaMemset(d_pix_mean, 0, gs_float2_size));
    CUDA_CHECK(cudaMemset(d_depth, 0, gs_float_size));
    CUDA_CHECK(cudaMemset(d_tiles_touched, 0, gs_uint32_size));

    int threads = std::min(256, static_cast<int>(gc));

    int blocks_preprocess = (gc + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    D_GS::preprocess<<<blocks_preprocess, threads>>>(
        FOCAL_X, FOCAL_Y, HEIGHT, WIDTH,
        d_gaussians, d_cov3Ds, d_cov2Ds,
        d_inv_cov, d_pix_mean, d_depth,
        d_tiles_touched,
        d_view_mat, d_proj_mat, gc);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    CUDA_KERNEL_CHECK();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Preprocess Time: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Prefix sum of tiles touched [8, 1, 2, 3] -> [8, 9, 11, 14]
    void *d_temp_storage = nullptr;
    // required size of Inclusive Sum
    size_t temp_storage_bytes_IS = 0;
    // first step get required size of temp_storage_bytes
    // first argument should be nullptr to determine that
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes_IS,
        d_tiles_touched, d_tiles_offset, gc
    ));

    // second step allocate required memory
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes_IS));

    // Finally run inlcusive prefix sum
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
       d_temp_storage, temp_storage_bytes_IS,
       d_tiles_touched, d_tiles_offset, gc
    ));

    // Last element of inclusive prefix sum contains the number of tiles in total
    // so lets copy only the last value
    uint32_t total_tiles = 0;
    CUDA_CHECK(cudaMemcpy(
       &total_tiles,
       &d_tiles_offset[gc-1],
       sizeof(uint32_t),
       cudaMemcpyDeviceToHost
    ));

    nvtx3::range_handle h2 = nvtx3::start_range("Second CudaMalloc batch", nvtx3::rgb{0,255,0});

    CUDA_CHECK(cudaMalloc((void**)&d_keys_unsorted, total_tiles * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_keys_sorted, total_tiles * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_values_unsorted, total_tiles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_values_sorted, total_tiles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_tiles_range, total_tiles * sizeof(uint2)));

    nvtx3::end_range(h2);

    // Get (unsorted) key-values arrays
    D_GS::gaussToTile<<<blocks_preprocess, threads>>>(
        CONSTANT::AABB_DIST,
        d_depth, d_cov2Ds, d_pix_mean,
        d_tiles_offset, d_keys_unsorted,
        d_values_unsorted, gc
    );

    CUDA_KERNEL_CHECK();

    // required size of Temp storege of radix sort
    size_t temp_storage_bytes_RS = 0;
    void *d_temp_storage_RS = nullptr;

    int tile_count_msb_idx = H_GS::get_msb_index(CONSTANT::GRID_X_MAX * CONSTANT::GRID_Y_MAX);
    tile_count_msb_idx += 1; // SortPairs expect values [beginBit, endBit) <--- !
    int end_bit = 32 + tile_count_msb_idx;

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
       nullptr, temp_storage_bytes_RS,
       d_keys_unsorted, d_keys_sorted,
       d_values_unsorted, d_values_sorted,
       total_tiles, 0, end_bit
    ));

    CUDA_CHECK(cudaMalloc((void**)&d_temp_storage_RS, temp_storage_bytes_RS));

    // Idea is, use exsting sort alrogithm from CUB
    // check how it works, then I will implement radix sort by myself

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
       d_temp_storage_RS, temp_storage_bytes_RS,
       d_keys_unsorted, d_keys_sorted,
       d_values_unsorted, d_values_sorted,
       total_tiles, 0, end_bit
    ));

    // To synchronize asynchronous Cuda operations
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_temp_storage));

    int blocks_tiles_size = (total_tiles + threads - 1) / threads;

    D_GS::identifyTileRanges<<<blocks_tiles_size, threads>>>(d_keys_sorted, d_tiles_range, total_tiles);
    CUDA_CHECK(cudaDeviceSynchronize());

    int blocks_render = (ele_num + threads - 1) / threads;

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, 0);

    int gridX = (CONSTANT::WIDTH + CONSTANT::TILE_X - 1) / CONSTANT::TILE_X;
    int gridY = (CONSTANT::HEIGHT + CONSTANT::TILE_Y - 1) / CONSTANT::TILE_Y;
    dim3 render_grid(gridX, gridY, 1);
    dim3 render_block(CONSTANT::TILE_X, CONSTANT::TILE_Y, 1);
    D_GS::render<<<render_grid, render_block>>>(
        CONSTANT::WIDTH, CONSTANT::HEIGHT,
        d_tiles_range, d_gaussians,
        d_pix_mean, d_inv_cov,
        d_values_sorted, d_image
    );
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    printf("Render Time: %f ms\n", milliseconds);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(img.data(), d_image, image_bytes, cudaMemcpyDeviceToHost));

    renderer.saveToPPM("image.ppm");

    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_gaussians));
    CUDA_CHECK(cudaFree(d_cov3Ds));
    CUDA_CHECK(cudaFree(d_cov2Ds));
    CUDA_CHECK(cudaFree(d_inv_cov));
    CUDA_CHECK(cudaFree(d_pix_mean));
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_tiles_touched));
    CUDA_CHECK(cudaFree(d_tiles_offset));
    CUDA_CHECK(cudaFree(d_keys_unsorted));
    CUDA_CHECK(cudaFree(d_keys_sorted));
    CUDA_CHECK(cudaFree(d_values_unsorted));
    CUDA_CHECK(cudaFree(d_values_sorted));
    CUDA_CHECK(cudaFree(d_tiles_range));
    return 0;
}
