#pragma once

#include <string>
#include <vector>
#include <fstream>

#include "ppm.hpp"

// H_GS - host gaussian splatting
namespace H_GS
{
    class Renderer
    {
        public:
            Renderer();

            void saveToPPM(const std::string& out_path);

            std::vector<pixel>& getImage() { return image; }
        private:
            std::vector<pixel> image;
    };
}
