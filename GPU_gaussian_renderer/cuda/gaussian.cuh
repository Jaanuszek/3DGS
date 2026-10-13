#pragma once

namespace D_GS
{
    // for now let's keep it as Array of structures
    // then profile it, check bottlenecks and improve it
    struct Gaussian{
        float3 mean3d;
        float3 scale;
        float4 rotation;
        float3 color;
        float opacity;
    };

    __global__ void preprocess();

    // Check if gaussian is visible by camera
    __device__ bool in_frustum(
        const float3& pos,
        const float3& viewMat
    );

    __device__ void computeCov3D(const float3& scale, const float4& rotation, float3* cov3D);

    __device__ void computeCov2D(const float3& mean);
}
