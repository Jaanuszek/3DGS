#include <cstdint>
#include <glm/detail/qualifier.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stdio.h>
#include <iostream>
#include <random>
#include <limits>

#include <curand_kernel.h>

#include "cuda/gaussian.cuh"
#include "include/defines.hpp"
#include "include/ppm.hpp"
#include "include/renderer.hpp"
#include "renderer.hpp"
#include "gaussian.cuh"

namespace
{
    constexpr uint32_t MIN = 0;
    constexpr uint32_t MAX = 255;

    constexpr uint32_t WIDTH = 800;
    constexpr uint32_t HEIGHT = 600;

    constexpr float ASPECT_RATIO = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);
    constexpr float NEAR = 0.1f;
    constexpr float FAR = 100.0f;
    constexpr float FOV = 90.0f;
    constexpr uint32_t GAUSSIANS_COUNT = 200;
    constexpr float AABB_DIST = 3;

    constexpr uint32_t PIXEL_COUNT = WIDTH * HEIGHT;

    constexpr uint32_t TILE_X = 16;
    constexpr uint32_t TILE_Y = 16;

    glm::vec3 cameraPos = glm::vec3(0.0f ,0.0f, 2.0f);
    glm::vec3 targetPos = glm::vec3(0.0f ,0.0f, 0.0f);
    glm::vec3 GLOBAL_UP = glm::vec3(0.0f, 1.0f, 0.0f);
}

using H_GS::pixel;
using D_GS::Gaussian;

__global__ void init_rand(curandState *state, unsigned long long seed)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

// kernel which randomly takes pixel color and save it to the image
__global__ void randomPixel(curandState *r_state, pixel* image, uint32_t image_size)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if(id >= image_size)
        return;

    // copy to not use global memory
    curandState localState = r_state[id];

    float rand_float_x = curand_uniform(&localState);
    float rand_float_y = curand_uniform(&localState);
    float rand_float_z = curand_uniform(&localState);

    int range = MAX - MIN + 1;
    int rand_int_x = MIN + (uint32_t)(rand_float_x * range);
    int rand_int_y = MIN + (uint32_t)(rand_float_y * range);
    int rand_int_z = MIN + (uint32_t)(rand_float_z * range);


    image[id] = pixel{
        (uint8_t)(rand_int_x), (uint8_t)(rand_int_y), (uint8_t)(rand_int_z)
    };
}

__global__ void renderGaussians(
    const glm::vec2* pix_mean, const float3* inv_cov,
    pixel* image, uint32_t image_size, uint32_t gc)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx >= image_size)
        return;

    int pix_x = idx % WIDTH;
    int pix_y = idx / WIDTH;
    glm::vec2 p(pix_x + 0.5f , pix_y + 0.5f);

    float r = 255.0f;
    float g = 255.0f;
    float b = 255.0f;

    for(int i = 0; i < gc; i++)
    {
        const glm::vec2 g_mean = pix_mean[i];
        const float3 g_inv_cov = inv_cov[i];
        glm::mat2 g_inv_cov_m = glm::mat2(
          g_inv_cov.x, g_inv_cov.y,
          g_inv_cov.y, g_inv_cov.z
        );

        const glm::vec2 d = p - g_mean;

        float q = glm::dot(d, g_inv_cov_m * d);

        float density = expf(-0.5f * q);
        r *= density;
        g *= density;
        b *= density;
    }

    image[idx] = pixel{
        static_cast<uint8_t>(r),
        static_cast<uint8_t>(g),
        static_cast<uint8_t>(b)
    };
}

int main()
{
    H_GS::Renderer renderer;

    std::random_device rd;
    std::mt19937 gen(rd());
    // std::uniform_int_distribution<unsigned long long> dist_ull(0, std::numeric_limits<unsigned long long>::max());

    Gaussian g1{
        .mean3d = glm::vec3(0.0f, 0.0f, -10.0f),
        .scale = glm::vec3(1.0f, 2.0f, 1.0f),
        .color = glm::vec3(1.0f, 1.0f, 1.0f),
        .rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        .opacity = 1.0f
    };

    std::vector<Gaussian> h_gaussians{
      g1
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
    curandState *d_rand_state = nullptr;

    Gaussian* d_gaussians = nullptr;
    float* d_cov3Ds = nullptr;
    float3* d_inv_cov = nullptr;
    glm::vec2* d_pix_mean = nullptr;
    glm::mat4* d_proj_mat = nullptr;
    glm::mat4* d_view_mat = nullptr;

    size_t ele_num = img.size();

    uint32_t image_size = ele_num * sizeof(pixel);

    cudaMalloc((void**)&d_image, image_size);
    cudaMalloc((void**)&d_rand_state, image_size);

    size_t gaussian_size = gc * sizeof(Gaussian);
    size_t cov3Ds_size = gc * sizeof(float);
    size_t inv_cov_size = gc * sizeof(float3);
    size_t pix_mean_size = gc * sizeof(glm::vec2);

    cudaMalloc((void**)&d_gaussians, gaussian_size);
    cudaMalloc((void**)&d_cov3Ds, cov3Ds_size);
    cudaMalloc((void**)&d_inv_cov, inv_cov_size);
    cudaMalloc((void**)&d_pix_mean, pix_mean_size);
    cudaMalloc((void**)&d_proj_mat, sizeof(glm::mat4));
    cudaMalloc((void**)&d_view_mat, sizeof(glm::mat4));

    cudaMemcpy(d_image, img.data(), image_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_gaussians, h_gaussians.data(), gaussian_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_proj_mat, glm::value_ptr(projMat), sizeof(glm::mat4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_view_mat, glm::value_ptr(viewMat), sizeof(glm::mat4), cudaMemcpyHostToDevice);
    cudaMemset(d_cov3Ds, 0, cov3Ds_size);
    cudaMemset(d_inv_cov, 0, inv_cov_size);
    cudaMemset(d_pix_mean, 0, pix_mean_size);

    int threads = 256;

    int blocks_preprocess = (gc + threads - 1) / threads;

    cudaError_t err = cudaGetLastError();

    D_GS::preprocess<<<blocks_preprocess, threads>>>(
        FOCAL_X, FOCAL_Y, HEIGHT, WIDTH,
        d_gaussians, d_cov3Ds,
        d_inv_cov, d_pix_mean,
        d_view_mat, d_proj_mat, gc);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "WTFFF " << cudaGetErrorString(err) << std::endl;
    }

    int blocks_render = (ele_num + threads - 1) / threads;

//     init_rand<<<blocks, threads>>>(d_rand_state, dist_ull(gen));
//
//     randomPixel<<<blocks, threads>>>(d_rand_state, d_image, ele_num);
    renderGaussians<<<blocks_render, threads>>>(
        d_pix_mean, d_inv_cov, d_image, image_size, gc
    );
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "WTFFF " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(img.data(), d_image, image_size, cudaMemcpyDeviceToHost);

    renderer.saveToPPM("image.ppm");

    cudaFree(d_image);
    cudaFree(d_rand_state);
    cudaFree(d_gaussians);
    cudaFree(d_cov3Ds);
    cudaFree(d_inv_cov);
    cudaFree(d_pix_mean);
    return 0;
}
