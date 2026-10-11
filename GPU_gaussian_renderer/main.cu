#include <stdio.h>
#include <iostream>
#include <random>
#include <limits>

#include <curand_kernel.h>

#include "include/defines.hpp"
#include "include/ppm.hpp"
#include "include/renderer.hpp"
#include "renderer.hpp"

constexpr uint32_t MIN = 0;
constexpr uint32_t MAX = 255;

using H_GS::pixel;

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

int main()
{
    H_GS::Renderer renderer;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned long long> dist_ull(0, std::numeric_limits<unsigned long long>::max());

    std::vector<pixel>& img = renderer.getImage();

    pixel* d_image = nullptr;
    curandState *d_rand_state = nullptr;

    size_t ele_num = img.size();

    uint32_t image_size = ele_num * sizeof(pixel);

    cudaMalloc((void**)&d_image, image_size);
    cudaMalloc((void**)&d_rand_state, image_size);

    cudaMemcpy(d_image, img.data(), image_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (ele_num + threads - 1) / threads;

    init_rand<<<blocks, threads>>>(d_rand_state, dist_ull(gen));

    randomPixel<<<blocks, threads>>>(d_rand_state, d_image, ele_num);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "WTFFF " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(img.data(), d_image, image_size, cudaMemcpyDeviceToHost);

    renderer.saveToPPM("image.ppm");

    cudaFree(d_image);
    cudaFree(d_rand_state);
    return 0;
}
