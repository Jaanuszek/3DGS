#ifndef HELPERS_CUH
#define HELPERS_CUH

namespace D_GS
{
    __device__ __forceinline__ float dot(float2 a, float2 b)
    {
        return a.x * b.x + a.y * b.y;
    }

    __device__ __forceinline__ float dot(float3 a, float3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __device__ __forceinline__ float3 transform_point3(const float*M, const float3& p)
    {
        float3 res = {
          M[0] * p.x +  M[4] * p.y + M[8] * p.z + M[12],
          M[1] * p.x +  M[5] * p.y + M[9] * p.z + M[13],
          M[2] * p.x +  M[6] * p.y + M[10] * p.z + M[14],
        };
        return res;
    }

    __device__ __forceinline__ float4 transform_point4(const float*M, const float3& p)
    {
        float4 res = {
          M[0] * p.x +  M[4] * p.y + M[8] * p.z + M[12],
          M[1] * p.x +  M[5] * p.y + M[9] * p.z + M[13],
          M[2] * p.x +  M[6] * p.y + M[10] * p.z + M[14],
          M[3] * p.x +  M[7] * p.y + M[11] * p.z + M[15],
        };
        return res;
    }

    __device__ __forceinline__ float4 transform_point4(const float* M, const float4& p)
    {
        float4 res = {
          M[0] * p.x +  M[4] * p.y + M[8] * p.z + M[12] * p.w,
          M[1] * p.x +  M[5] * p.y + M[9] * p.z + M[13] * p.w,
          M[2] * p.x +  M[6] * p.y + M[10] * p.z + M[14] * p.w,
          M[3] * p.x +  M[7] * p.y + M[11] * p.z + M[15] * p.w,
        };
        return res;
    }


    __device__ __forceinline__ float2 mat2_mul_vec2(const float* M, float2 v)
    {
        return make_float2(
          M[0] * v.x + M[2] * v.y,
          M[1] * v.x + M[3] * v.y
        );
    }

    __device__ __forceinline__ float3 mat4_mul_vec3(const float*M, float3 v)
    {
        float3 p = {
          M[0] * v.x +  M[4] * v.y + M[8] * v.z,
          M[1] * v.x +  M[5] * v.y + M[9] * v.z,
          M[2] * v.x +  M[6] * v.y + M[10] * v.z,
        };
        return p;
    }

    __device__ __forceinline__ float4 mat4_mul_vec4(const float*M, float4 v)
    {
        float4 p = {
          M[0] * v.x +  M[4] * v.y + M[8] * v.z,
          M[1] * v.x +  M[5] * v.y + M[9] * v.z,
          M[2] * v.x +  M[6] * v.y + M[10] * v.z,
          M[3] * v.x +  M[7] * v.y + M[11] * v.z,
        };
        return p;
    }

    __device__ __forceinline__ int clamp(int f, int a, int b)
    {
        return max(a, min(f, b));
    }

    __device__ __forceinline__ int2 clamp(int2 v, int a, int b)
    {
        return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
    }

    __device__ __forceinline__ int2 clamp(int2 v, int2 a, int2 b)
    {
        return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
    }
}

#endif
