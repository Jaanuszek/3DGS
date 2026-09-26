#pragma once

#include <cstdint>

struct pixel{
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;

    pixel operator+(const pixel& other) const{
        return pixel(r + other.r, g + other.g, b + other.b);
    }

    pixel& operator+=(const pixel& rhs)
    {
        // SO basically it may introduce bugs, since C++ by default
        // promotes std::uint8_t to int during add operation
        // so if this->r += rhs.r is larget than 255 then the result will be clipped
        // to value & 0xFF
        this->r += rhs.r;
        this->g += rhs.g;
        this->b += rhs.b;
        return *this;
    }
};
