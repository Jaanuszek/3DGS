#pragma once

#include <vector>
#include <atomic>
#include <string>
#include <fstream>
#include <algorithm>
#include "defines.hpp"
#include "ppm.hpp"
#include "glm/glm.hpp"

#include "world.hpp"


#include <iostream>

namespace GS
{
    class Renderer
    {
        public:
            Renderer(World&& world);

            void render_tile(uint32_t tile_id);

            void saveToPPM(const std::string& out_path);
        private:
            void gaussToTile();

            // Sort gaussians in tiles by their depth
            void sortTiles();

        private:
            std::vector<pixel> image;
            std::vector<std::vector<uint32_t>> Tiles;

            World world;

            // uint32_t tile_id{0};

            uint32_t GRID_X_MAX;
            uint32_t GRID_Y_MAX;
    };
}
