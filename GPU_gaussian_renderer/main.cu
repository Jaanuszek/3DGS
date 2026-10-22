#include <cstdint>
#include <glm/detail/qualifier.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdio.h>
#include <iostream>
#include <random>
#include <limits>

#include <curand_kernel.h>
#include "cuda/helpers.cuh"
#include "cuda_check.cuh"

#include "cuda/gaussian.cuh"
#include "include/defines.hpp"
#include "include/ppm.hpp"
#include "include/renderer.hpp"
#include "renderer.hpp"
#include "gaussian.cuh"

namespace
{
    // constexpr uint32_t MIN = 0;
    constexpr uint32_t MAX = 255;
//
    constexpr uint32_t WIDTH = 800;
    constexpr uint32_t HEIGHT = 600;

    constexpr float ASPECT_RATIO = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);
    constexpr float NEAR = 0.1f;
    constexpr float FAR = 100.0f;
    constexpr float FOV = 90.0f;
    // constexpr uint32_t GAUSSIANS_COUNT = 200;
    // constexpr float AABB_DIST = 3;

    // constexpr uint32_t PIXEL_COUNT = WIDTH * HEIGHT;

    // constexpr uint32_t TILE_X = 16;
    // constexpr uint32_t TILE_Y = 16;

    glm::vec3 cameraPos = glm::vec3(0.0f ,0.0f, 2.0f);
    glm::vec3 targetPos = glm::vec3(0.0f ,0.0f, 0.0f);
    glm::vec3 GLOBAL_UP = glm::vec3(0.0f, 1.0f, 0.0f);
}

using H_GS::pixel;
using D_GS::Gaussian;

__global__ void renderGaussians(
    const D_GS::Gaussian* gaussians,
    const float2* pix_mean, const float3* inv_cov,
    pixel* image, uint32_t image_size, uint32_t gc)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx >= image_size)
        return;

    int pix_x = idx % WIDTH;
    int pix_y = idx / WIDTH;
    glm::vec2 p(pix_x + 0.5f , pix_y + 0.5f);

    glm::vec3 C(0.0f);
    float T = 1.0f;

    for(int i = 0; i < gc; i++)
    {
        const float2 g_mean = pix_mean[i];
        const float3 g_inv_cov = inv_cov[i];
        const Gaussian g = gaussians[i];

        glm::mat2 g_inv_cov_m = glm::mat2(
          g_inv_cov.x, g_inv_cov.y,
          g_inv_cov.y, g_inv_cov.z
        );

        float* inv_cov_m_p = glm::value_ptr(g_inv_cov_m);

        const float2 d = make_float2(
            p.x - g_mean.x,
            p.y - g_mean.y
        );

        // float q = glm::dot(d, g_inv_cov_m * d);
        float q = D_GS::dot(d, D_GS::mat2_mul_vec2(inv_cov_m_p, d));

        float density = expf(-0.5f * q);

        float alpha = min(0.99f, density * g.opacity);

        if(alpha < 1.0f / 255.0f)
            continue;

        // This does not work in kernel:
        // C += g.color * (T * alpha);
        const float cr = g.color.x;
        const float cg = g.color.y;
        const float cb = g.color.z;

        const float weight = T * alpha;

        C.x += cr * weight;
        C.y += cg * weight;
        C.z += cb * weight;
        T *= (1.0f - alpha);
        if(T < 0.0001f)
            break;
    }
    C = glm::clamp(C,0.0f, 1.0f);

    image[idx].r = static_cast<uint8_t>(C.r * 255.0f);
    image[idx].g = static_cast<uint8_t>(C.g * 255.0f);
    image[idx].b = static_cast<uint8_t>(C.b * 255.0f);
}

int main()
{
    H_GS::Renderer renderer;

    D_GS::Gaussian g1{
        .rotation = make_float4(1.0f, 0.0f, 0.0f, 0.0f),
        .mean3d = make_float3(0.0f, 0.0f, -10.0f),
        .scale = make_float3(1.0f, 2.0f, 1.0f),
        .color = make_float3(0.3f, 1.0f, 0.08),
        .opacity = 0.5f
    };

    D_GS::Gaussian g2{
        .rotation = make_float4(0.70710678f, 0.0f, 0.70710678f, 0.0f),
        .mean3d = make_float3(1.0f, 1.0f, -10.0f),
        .scale = make_float3(1.0f, 1.0f, 1.0f),
        .color = make_float3(1.0f, 0.0f, 0.0f),
        .opacity = 1.0f
    };

    D_GS::Gaussian g3{
        .rotation = make_float4(1.0f, 0.0f, 0.0f, 0.0f),
        .mean3d = make_float3(-2.0f, -2.0f, -10.0f),
        .scale = make_float3(3.0f, 1.0f, 5.0f),
        .color = make_float3(0.0f, 0.0f, 1.0f),
        .opacity = 0.7f
    };

    std::vector<Gaussian> h_gaussians{
      g1, g2, g3
    };

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

    std::vector<pixel>& img = renderer.getImage();

    pixel* d_image = nullptr;

    Gaussian* d_gaussians = nullptr;
    float* d_cov3Ds = nullptr;
    float3* d_inv_cov = nullptr;
    float2* d_pix_mean = nullptr;
    float* d_proj_mat = nullptr;
    float* d_view_mat = nullptr;

    size_t ele_num = img.size();

    uint32_t image_size = static_cast<uint32_t>(ele_num);
    size_t image_bytes = ele_num * sizeof(pixel);

    CUDA_CHECK(cudaMalloc((void**)&d_image, image_bytes));

    size_t gaussian_size = gc * sizeof(D_GS::Gaussian);
    size_t cov3Ds_size = gc * sizeof(float);
    size_t inv_cov_size = gc * sizeof(float3);
    size_t pix_mean_size = gc * sizeof(float2);
    size_t mat4x4_size_f = 16 * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_gaussians, gaussian_size));

    CUDA_CHECK(cudaMalloc((void**)&d_cov3Ds, cov3Ds_size));
    CUDA_CHECK(cudaMalloc((void**)&d_inv_cov, inv_cov_size));
    CUDA_CHECK(cudaMalloc((void**)&d_pix_mean, pix_mean_size));
    CUDA_CHECK(cudaMalloc((void**)&d_proj_mat, mat4x4_size_f));
    CUDA_CHECK(cudaMalloc((void**)&d_view_mat, mat4x4_size_f));

    CUDA_CHECK(cudaMemcpy(d_image, img.data(), image_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_gaussians, h_gaussians.data(), gaussian_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_proj_mat, glm::value_ptr(projMat), mat4x4_size_f, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_view_mat, glm::value_ptr(viewMat), mat4x4_size_f, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_cov3Ds, 0, cov3Ds_size));
    CUDA_CHECK(cudaMemset(d_inv_cov, 0, inv_cov_size));
    CUDA_CHECK(cudaMemset(d_pix_mean, 0, pix_mean_size));

    int threads = std::min(256, static_cast<int>(gc));

    int blocks_preprocess = (gc + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    D_GS::preprocess<<<blocks_preprocess, threads>>>(
        FOCAL_X, FOCAL_Y, HEIGHT, WIDTH,
        d_gaussians, d_cov3Ds,
        d_inv_cov, d_pix_mean,
        d_view_mat, d_proj_mat, gc);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Preprocess Time: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CUDA_KERNEL_CHECK();

    int blocks_render = (ele_num + threads - 1) / threads;

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, 0);
    renderGaussians<<<blocks_render, threads>>>(
        d_gaussians,
        d_pix_mean,
        d_inv_cov,
        d_image,
        static_cast<uint32_t>(image_size),
        static_cast<uint32_t>(gc)
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
    CUDA_CHECK(cudaFree(d_inv_cov));
    CUDA_CHECK(cudaFree(d_pix_mean));
    return 0;
}
