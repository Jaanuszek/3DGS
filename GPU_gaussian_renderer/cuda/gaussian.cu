#include "gaussian.cuh"
// #include <__clang_cuda_builtin_vars.h>

namespace D_GS
{
    __global__ void preprocess(
        Gaussian* gaussians,
        float* cov3Ds,
        const glm::mat4 *viewMat,
        const glm::mat4 *projMat,
        const int gc)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        if(idx >= gc)
            return;

        Gaussian gs = gaussians[idx];

        // 0. Multiply gaussian mean by proj and view matrix
        // 1.Check if point is in frustum if not return early
        // 2. compute 3D covariance
        // 3. Compute Jacobian
        // 4. Compute 2D covariance (in view space)

        // 0.
        glm::vec4 pointCam = *viewMat * glm::vec4(gs.mean3d, 1.0f);
        glm::vec4 pointProj = *projMat * pointCam;

        // 1.
        if(!in_frustum(gs.mean3d, *viewMat, *projMat))
            return;

        //2.
        float* cov3D = cov3Ds + idx * 6;
        computeCov3D(gs.scale, gs.rotation, float *cov3D)
    }

    __device__ bool in_frustum(
        const glm::vec3& pos,
        const glm::mat4& viewMat,
        const glm::mat4& projMat
    )
    {
        // 0. Transfer gauss pos from world cords to clip space
        //      0.5 cameraSpace = pos * viewMat -> clipSpace = cameraSpace * projMat
        // 1. check if point lies in  frustum
        // 1.1 |xc| <= wc
        // 1.2 |yc| <= wc
        // 1.3 |zc| <= wc
        // 1.4 wc > 0

        // glm is column - major, therefore MAT * POINT is a proper order
        glm::vec4 pointCam = viewMat * glm::vec4(pos, 1.0f);
        glm::vec4 pointProj = projMat * pointCam;

        bool isVisible = pointProj.w > 0.0f && // to not divide by zero in perspective division
            pointProj.z >= -pointProj.w && // near
            pointProj.z <= pointProj.w && // far
            pointProj.x >= -pointProj.w && pointProj.x <= pointProj.w &&
            pointProj.y >= -pointProj.w && pointProj.y <= pointProj.w;

        if(isVisible)
        {
            return true;
        }

        return false;
    }

    __device__ void computeCov3D(const glm::vec3& scale, const glm::quat& rotation, float* cov3D)
    {
        glm::mat3 scale_m = glm::mat3(1.0f);
        scale_m[0][0] = scale.x;
        scale_m[1][1] = scale.y;
        scale_m[2][2] = scale.z;

        glm::mat3 rotation_m = glm::mat3_cast(glm::normalize(rotation));

        glm::mat3 Sigma = rotation_m * scale_m * glm::transpose(scale_m) * glm::transpose(rotation_m);

        cov3D[0] = Sigma[0][0];
        cov3D[1] = Sigma[0][1];
        cov3D[2] = Sigma[0][2];
        cov3D[3] = Sigma[1][1];
        cov3D[4] = Sigma[1][2];
        cov3D[5] = Sigma[2][2];
    }

    __device__ void computeCov2D(const float3& mean)
    {

    }
}
