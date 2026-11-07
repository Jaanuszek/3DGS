#pragma once

#include <cstdint>

namespace H_GS
{
    struct pixel{
        uint8_t r;
        uint8_t g;
        uint8_t b;

        pixel operator+(const pixel& other) const{
            return pixel(r + other.r, g + other.g, b + other.b);
        }

        pixel& operator+=(const pixel& rhs)
        {
            this->r += rhs.r;
            this->g += rhs.g;
            this->b += rhs.b;
            return *this;
        }

    };
}
