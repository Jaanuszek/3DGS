#pragma once

#include <vector>
#include <atomic>
#include <string>
#include <fstream>
#include "defines.hpp"
#include "ppm.hpp"
#include "glm/glm.hpp"

namespace GS
{
    class Renderer
    {
        public:
            Renderer(uint32_t grid_max_x, uint32_t grid_max_y);

            void render_tile(std::atomic<uint32_t> tile_id);

            void saveToPPM(const std::string& out_path);
        private:
            std::vector<pixel> image;
            std::vector<glm::vec3> colorBuffer;
            std::vector<float> transmittanceBuffer;
            std::vector<std::vector<uint32_t>> Tiles;
    }
}
