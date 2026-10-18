#pragma once

#include "glm/glm.hpp"
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
        Gaussian* gaussians,
        float* cov3Ds,
        const glm::mat4 *viewMat,
        const glm::mat4 *projMat,
        const int gc
    );

    // Check if gaussian is visible by camera
    __device__ bool in_frustum(
        const glm::vec3& pos,
        const glm::mat4& viewMat,
        const glm::mat4& projMat
    );

    __device__ void computeCov3D(const glm::vec3& scale, const glm::quat& rotation, float* cov3D);

    __device__ void computeCov2D(const float3& mean);
}
