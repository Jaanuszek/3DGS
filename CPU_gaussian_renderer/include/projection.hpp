#pragma once

#include <iostream>
#include <array>

// 1st my own impl
namespace My
{
    template<typename T, std::size_t N>
    struct vec
    {
        std::array<T,N> data;

        T& operator[](std::size_t idx)
        {
            return data[idx];
        }

        const T& operator[](std::size_t idx) const
        {
            return data[idx];
        }

        T& x() requires (N >= 1)
        {
            return data[0];
        }
        T& y() requires (N >= 2)
        {
            return data[1];
        }
        T& z() requires (N >=3)
        {
            return data[2];
        }
        T& w() requires (N >= 4)
        {
            return data[3];
        }
        // T& operator*()
    };
    // N - rows
    // M - columns
    template <typename T, std::size_t N, std::size_t M>
    struct mat{
        // std::array<std::array<T,M>,N> data;
        std::array<vec<T, M>, N> data;

        T& operator()(std::size_t n, std::size_t m)
        {
            return data[n][m];
        }

        const T& operator()(std::size_t n, std::size_t m) const
        {
            return data[n][m];
        }

        // void mul_v(const vec<T, >)
        // T& operator*(const vec<T,N>& vec)
        // {
            // vec<T,N> temp{};
            //
        // }

        void print()
        {
            for(int n = 0; n < N; n++)
            {
                for(int m = 0; m < M; m++)
                {
                    std::cout << operator()(n,m) << " ";
                }
                std::cout << std::endl;
            }
        }
    };

    using mat4x4_f = mat<float, 4, 4>;
    using vec3_f = vec<float, 3>;

    constexpr float PI = std::numbers::pi;

    mat4x4_f createProjectionMatrix(
        const float& fov,
        const float& near,
        const float& far
    );

    vec3_f mulPointMatrix(const vec3_f& in, const mat4x4_f& mat);
}
