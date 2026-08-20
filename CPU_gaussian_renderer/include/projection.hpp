#pragma once

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
    };
    // N - rows
    // M - columns
    template <typename T, std::size_t N, std::size_t M>
    struct mat{
        std::array<std::array<T,M>,N> data;

        T& operator()(std::size_t n, std::size_t m)
        {
            return data[n][m];
        }
        // void mul(const vec4<T>& vec);
    };
}
