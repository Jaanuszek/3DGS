#include "gaussian.cuh"
#include <glm/matrix.hpp>
// #include <__clang_cuda_builtin_vars.h>

namespace D_GS
{
    __global__ void preprocess(
        const float f_x, const float f_y,
        const uint32_t H, const uint32_t W,
        Gaussian* gaussians,
        float* cov3Ds,
        float3* inv_sigma,
        glm::vec2* pix_mean,
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
        // 5. Compute inverse sigma (to be able to compute final gaussian density)
        // 6. compute projected mean to pixel
        // 6.5 change projected point from homogenous to cartesian cords
        // 7. Store necessary data in global memory
        // 7.1 invSigma (inv 2D covariance mat)
        // 7.2 mean_pixel (as float2 -> x,y)

        // 0.
        glm::vec4 pointCam = *viewMat * glm::vec4(gs.mean3d, 1.0f);
        glm::vec4 pointProj = *projMat * pointCam;

        // 1.
        if(!in_frustum(gs.mean3d, *viewMat, *projMat))
            return;

        // 2.
        float* cov3D = cov3Ds + idx * 6;
        computeCov3D(gs.scale, gs.rotation, cov3D);

        // 3, 4.
        float3 cov2D = computeCov2D(
            gs.mean3d, cov3D,
            *viewMat, f_x, f_y
        );

        // 5.
        // it actually stores result in global memory
        float3* cov_inv = inv_sigma + idx;
        computeInvSigma(cov2D, *cov_inv);

        // 6.5
        // p_proj is in cartesian cords (3D not 4D)
        glm::vec3 p_proj = hom2Cart(pointProj);

        // 6.
        glm::vec2 mean_pix = glm::vec2(
          ndc2Pix(p_proj.x, W),
          ndc2Pix_inv(p_proj.y, H)
        );

        //7.
        pix_mean[idx] = mean_pix;
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

    __device__ float3 computeCov2D(const glm::vec3& mean, const float* cov3D, const glm::mat4& view_m, float f_x, float f_y)
    {
        // I might add here bigger fov, so gaussian tat has center near camera
        // frustum can be rendered. See computeCov2D here https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu#L74
        glm::vec4 pointCam = view_m * glm::vec4(mean, 1.0f);

        float z_2 = pointCam.z * pointCam.z;
        glm::mat3x2 J = glm::mat3x2(
          f_x / pointCam.z, 0.0f,
          0, f_y / pointCam.z,
          (-f_x * pointCam.x) / (z_2), (-f_y * pointCam.y) / (z_2)
        );

        // rotation part of view matrix
        glm::mat3 W = glm::mat3(view_m);

        glm::mat3 cov3d_mat = glm::mat3(
            cov3D[0], cov3D[1], cov3D[2],
            cov3D[1], cov3D[3], cov3D[4],
            cov3D[2], cov3D[4], cov3D[5]
        );

        glm::mat3 T = W * cov3d_mat * glm::transpose(W);
        glm::mat2 cov2D = J * T * glm::transpose(J);

        return { float(cov2D[0][0]), float(cov2D[0][1]), float(cov2D[1][1]) };
    }

    __device__ void computeInvSigma(const float3& cov2D, float3& inv_sigma)
    {
        float a = cov2D.x; // [0][0]
        float b = cov2D.y; // [0][1]
        float c = cov2D.z; // [1][1]

        float det = a * c - b * b;
        if(det <= 1e-8f)
        {
            inv_sigma = make_float3(0.0f, 0.0f, 0.0f);
            return;
        }

        float det_inv = 1 / det;

       inv_sigma.x = det_inv * c;
       inv_sigma.y = -det_inv * b;
       inv_sigma.z = det_inv * a;
    }

    __device__ glm::vec3 hom2Cart(const glm::vec4& p_hom)
    {
        float p_w = 1.0f / p_hom.w;
        return glm::vec3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w);
    }

    __device__ float ndc2Pix(const float v, const float S)
    {
        return ((v + 1.0f) * S - 1.0f) * 0.5f;
    }

    __device__ float ndc2Pix_inv(const float v, const float S)
    {
        return ((1.0f - v) * S - 1.0f) * 0.5f;
    }
}
