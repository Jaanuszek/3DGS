#include "gaussian.cuh"
#include <__clang_cuda_builtin_vars.h>

namespace D_GS
{
    __global__ void preprocess(
        Gaussian* gaussians,
        const glm::mat4 *viewMat,
        const glm::mat4 *projMat,
        const int gc)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        if(idx >= gc)
            return;

        // 0. Multiply gaussian mean by proj and view matrix
        // 1.Check if point is in frustum if not return early
        // 2. compute 3D covariance
        // 3. Compute Jacobian
        // 4. Compute 2D covariance (in view space)
    }

    __device__ bool in_frustum(
        const float3& pos,
        const float3& viewMat
    )
    {

    }

    __device__ void computeCov3D(const float3& scale, const float4& rotation, float3* cov3D)
    {

    }

    __device__ void computeCov2D(const float3& mean)
    {

    }
}
