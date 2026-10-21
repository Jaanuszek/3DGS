#pragma once

#include <glm/glm.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/quaternion_common.hpp>
#include "glm/gtc/quaternion.hpp"

namespace D_GS
{
    // for now let's keep it as Array of structures
    // then profile it, check bottlenecks and improve it
    struct Gaussian{
        glm::vec3 mean3d;
        glm::vec3 scale;
        glm::vec3 color;
        glm::quat rotation;
        float opacity;
    };

    // gc - gaussian count
    __global__ void preprocess(
        const float f_x, const float f_y,
        const uint32_t H, const uint32_t W,
        Gaussian* gaussians,
        float* cov3Ds,
        float3* inv_sigma,
        glm::vec2* pix_mean,
        const glm::mat4 *viewMat,
        const glm::mat4 *projMat,
        const int gc);

    // Check if gaussian is visible by camera
    __device__ bool in_frustum(
        const glm::vec3& pos,
        const glm::mat4& viewMat,
        const glm::mat4& projMat
    );

    __device__ void computeCov3D(const glm::vec3& scale, const glm::quat& rotation, float* cov3D);

    // returning float3 cov2D because I dont need to save it anywhere in memory
    // It will be recalculated every frame anyway
    __device__ float3 computeCov2D(const glm::vec3& mean, const float* cov3D, const glm::mat4& view_m, float f_x, float f_y);

    __device__ void computeInvSigma(const float3& cov2D, float3& inv_sigma);

    __device__ glm::vec3 hom2Cart(const glm::vec4& p_hom);

    __device__ float ndc2Pix(const float v, const float S);

    __device__ float ndc2Pix_inv(const float v, const float S);
}
