#pragma once

#include <cstdint>

namespace  CONSTANT{
    constexpr int WIDTH = 800;
    constexpr int HEIGHT = 600;
    constexpr float FOV = 90.0f;
    constexpr uint32_t GAUSSIANS_COUNT = 200;
    constexpr float AABB_DIST = 3;

    constexpr uint32_t TILE_X = 16;
    constexpr uint32_t TILE_Y = 16;
}
