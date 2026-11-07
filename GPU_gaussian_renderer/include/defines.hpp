#pragma once

#include <cstdint>

namespace  CONSTANT{
    constexpr uint32_t WIDTH = 800;
    constexpr uint32_t HEIGHT = 600;
    constexpr float FOV = 90.0f;
    constexpr uint32_t GAUSSIANS_COUNT = 200;
    constexpr float AABB_DIST = 3;

    constexpr uint32_t PIXEL_COUNT = WIDTH * HEIGHT;

    constexpr uint32_t TILE_X = 16;
    constexpr uint32_t TILE_Y = 16;

    constexpr int GRID_X_MAX = (WIDTH + TILE_X - 1) / TILE_X;
    constexpr int GRID_Y_MAX = (HEIGHT + TILE_Y -1) / TILE_Y;
}
