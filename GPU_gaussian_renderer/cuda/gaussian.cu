#include "gaussian.cuh"
#include "helpers.cuh"
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
        float2* pix_mean,
        float* depth,
        uint32_t* tiles_touched,
        const float *viewMat,
        const float *projMat,
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
        // 8. Compute and save number of tiles touched by this particular gaussian

        // 0.
        // glm::vec4 pointCam = *viewMat * glm::vec4(gs.mean3d, 1.0f);
        // glm::vec4 pointProj = *projMat * pointCam;
        float4 pointCam = transform_point4(viewMat, gs.mean3d);
        float4 pointProj = transform_point4(projMat, pointCam);

        tiles_touched[idx] = 0;


        // 1.
        if(!in_frustum(gs.mean3d, viewMat, projMat))
            return;

        // 2.
        float* cov3D = cov3Ds + idx * 6;
        computeCov3D(gs.scale, gs.rotation, cov3D);

        // 3, 4.
        float3 cov2D = computeCov2D(
            gs.mean3d, cov3D,
            viewMat, f_x, f_y
        );

        // 5.
        // it actually stores result in global memory
        float3* cov_inv = inv_sigma + idx;
        computeInvSigma(cov2D, *cov_inv);

        // 6.5
        // p_proj is in cartesian cords (3D not 4D)
        float3 p_proj = hom2Cart(pointProj);

        // 6.
        float2 mean_pix = make_float2(
          ndc2Pix(p_proj.x, W),
          ndc2Pix_inv(p_proj.y, H)
        );

        // 8.
        TileRange t_range = getRect(CONSTANT::AABB_DIST, cov2D, mean_pix);
        uint32_t t_touched = (t_range.t_max.x - t_range.t_min.x) * (t_range.t_max.y - t_range.t_min.y);
        if(t_touched == 0)
            return; // return before saving anything to memory since this gaussian will be irrelevant

        //7.
        depth[idx] = pointCam.z;
        pix_mean[idx] = mean_pix;
        tiles_touched[idx] = t_touched;
    }

    __device__ bool in_frustum(
        const float3& pos,
        const float* viewMat,
        const float* projMat
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
        float4 pointCam = transform_point4(viewMat, pos);
        float4 pointProj = transform_point4(projMat, pointCam);

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

    __device__ void computeCov3D(const float3& scale, const float4& rotation, float* cov3D)
    {
        glm::mat3 scale_m = glm::mat3(1.0f);
        scale_m[0][0] = scale.x;
        scale_m[1][1] = scale.y;
        scale_m[2][2] = scale.z;

        float4 q = rotation;
        const float w = q.x;
        const float x = q.y;
        const float y = q.z;
        const float z = q.w;

        // TODO normalize this quaternion
        glm::mat3 R = glm::mat3(
            1.0f - 2.0f * (y*y + z*z), 2.0f*(x*y + z*w), 2*(x*z - y*w),
            2.0f*(x*y - z*w), 1.0f-2.0f*(x*x + z*z), 2*(y*z+x*w),
            2.0f*(x*z + y*w), 2*(y*z - x*w), 1.0f - 2.0f*(x*x + y*y)
        );

        glm::mat3 Sigma = R * scale_m * glm::transpose(scale_m) * glm::transpose(R);

        cov3D[0] = Sigma[0][0];
        cov3D[1] = Sigma[0][1];
        cov3D[2] = Sigma[0][2];
        cov3D[3] = Sigma[1][1];
        cov3D[4] = Sigma[1][2];
        cov3D[5] = Sigma[2][2];
    }

    __device__ float3 computeCov2D(const float3& mean, const float* cov3D, const float* view_m, float f_x, float f_y)
    {
        // I might add here bigger fov, so gaussian tat has center near camera
        // frustum can be rendered. See computeCov2D here https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu#L74
        float4 pointCam = transform_point4(view_m, mean);

        float z_2 = pointCam.z * pointCam.z;
        glm::mat3x2 J = glm::mat3x2(
          f_x / pointCam.z, 0.0f,
          0, f_y / pointCam.z,
          (-f_x * pointCam.x) / (z_2), (-f_y * pointCam.y) / (z_2)
        );

        // rotation part of view matrix
        // In case of issues check if this indexing is correct!
        glm::mat3 W = glm::mat3(
            view_m[0], view_m[1], view_m[2],
            view_m[4], view_m[5], view_m[6],
            view_m[8], view_m[9], view_m[10]
        );

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

    __device__ float3 hom2Cart(const float4& p_hom)
    {
        float p_w = 1.0f / p_hom.w;
        return make_float3(p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w);
    }

    __device__ float ndc2Pix(const float v, const float S)
    {
        return ((v + 1.0f) * S - 1.0f) * 0.5f;
    }

    __device__ float ndc2Pix_inv(const float v, const float S)
    {
        return ((1.0f - v) * S - 1.0f) * 0.5f;
    }

    __device__ TileRange getRect(float dist, float3 cov2D, float2 mean_pix)
    {
        float a = cov2D.x;
        float d = cov2D.z;

        float rx = dist * sqrt(a);
        float ry = dist * sqrt(d);

        const int min_x = static_cast<int>(
            floor((mean_pix.x - rx) / CONSTANT::TILE_X)
        );
        const int max_x = static_cast<int>(
            ceil((mean_pix.x + rx) / CONSTANT::TILE_X)
        );
        const int min_y = static_cast<int>(
            floor((mean_pix.y - ry) / CONSTANT::TILE_Y)
        );
        const int max_y = static_cast<int>(
            ceil((mean_pix.y + ry) / CONSTANT::TILE_Y)
        );

        int2 tile_min = make_int2(min_x, min_y);
        int2 tile_max = make_int2(max_x, max_y);
        int2 range_min = make_int2(0,0);
        int2 range_max = make_int2(CONSTANT::GRID_X_MAX, CONSTANT::GRID_Y_MAX);

        return TileRange{
            .t_min = clamp(tile_min, range_min, range_max),
            .t_max = clamp(tile_max, range_min, range_max)
        };
    }
}
