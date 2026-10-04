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

namespace GS
{
    class Renderer
    {
        public:
            Renderer(World&& world);

            void render_tile();

            void saveToPPM(const std::string& out_path);
        private:
            void gaussToTile();

            // Sort gaussians in tiles by their depth
            void sortTiles();

        private:
            std::vector<pixel> image;
            std::vector<glm::vec3> colorBuffer;
            std::vector<float> transmittanceBuffer;
            std::vector<std::vector<uint32_t>> Tiles;

            World world;

            std::atomic<uint32_t> tile_id{};

            uint32_t GRID_X_MAX;
            uint32_t GRID_Y_MAX;
    };
}
