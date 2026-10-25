#pragma once

#include "defines.hpp"

#include <glm/glm.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/quaternion_common.hpp>
#include "glm/gtc/quaternion.hpp"

namespace D_GS
{
    // for now let's keep it as Array of structures
    // then profile it, check bottlenecks and improve it
    struct Gaussian{
        float4 rotation;
        float3 mean3d;
        float3 scale;
        float3 color;
        float opacity;
    };

    struct TileRange
    {
        int2 t_min;
        int2 t_max;
    };

    // gc - gaussian count
    __global__ void preprocess(
        const float f_x, const float f_y,
        const uint32_t H, const uint32_t W,
        Gaussian* gaussians,
        float* cov3Ds,
        float3* inv_sigma,
        float2* pix_mean,
        float* depth,
        uint32_t* tiles_touched,
        const float *viewMat,
        const float *projMat,
        const int gc);

    // Check if gaussian is visible by camera
    __device__ bool in_frustum(
        const float3& pos,
        const float* viewMat,
        const float* projMat
    );

    __device__ void computeCov3D(const float3& scale, const float4& rotation, float* cov3D);

    // returning float3 cov2D because I dont need to save it anywhere in memory
    // It will be recalculated every frame anyway
    __device__ float3 computeCov2D(const float3& mean, const float* cov3D, const float* view_m, float f_x, float f_y);

    __device__ void computeInvSigma(const float3& cov2D, float3& inv_sigma);

    __device__ float3 hom2Cart(const float4& p_hom);

    __device__ float ndc2Pix(const float v, const float S);

    __device__ float ndc2Pix_inv(const float v, const float S);

    __device__ TileRange getRect(float dist, float3 cov2D, float2 mean_pix);
}
