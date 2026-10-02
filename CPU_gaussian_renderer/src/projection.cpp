#include "projection.hpp"
#include <cmath>
namespace My
{
    mat4x4_f createProjectionMatrix(
        const float& fov,
        const float& near,
        const float& far
    )
    {
        mat4x4_f mat;
        float S = 1 / (tan((fov/2)*(PI/180)));
        mat(0,0) = S;
        mat(1,1) = S;
        mat(2,2) = -(far/(far - near));
        mat(2,3) = -1;
        mat(3,2) = -((far * near)/ (far-near));
        return mat;
    }

    vec3_f mulPointMatrix(const vec3_f& in, const mat4x4_f& mat4x4)
    {
        vec3_f vec;
        // Assuming w = 1
        vec[0] = in[0] * mat4x4(0,0) + in[1] * mat4x4(0,1) + in[2] * mat4x4(0,2) + mat4x4(0,3);
        vec[1] = in[0] * mat4x4(1,0) + in[1] * mat4x4(1,1) + in[2] * mat4x4(1,2) + mat4x4(1,3);
        vec[2] = in[0] * mat4x4(2,0) + in[1] * mat4x4(2,1) + in[2] * mat4x4(2,2) + mat4x4(2,3);
        float w = in[0] * mat4x4(3,0) + in[1] * mat4x4(3,1) + in[2] * mat4x4(3,2) + mat4x4(3,3);

        if (w != 1)
        {
            vec[0] /= w;
            vec[1] /= w;
            vec[2] /= w;
        }

        return vec;
    }
}
