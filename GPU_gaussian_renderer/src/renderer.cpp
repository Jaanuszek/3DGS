#include "renderer.hpp"
#include "defines.hpp"

namespace H_GS
{
    Renderer::Renderer()
    {
        image.resize(CONSTANT::PIXEL_COUNT);
    }

    void Renderer::saveToPPM(const std::string &out_path)
    {
        std::ofstream file(out_path, std::ios::binary);
        if(!file)
            return;

        file << "P6\n";
        file << CONSTANT::WIDTH << " " << CONSTANT::HEIGHT << '\n';
        file << "255\n";

        file.write(
            reinterpret_cast<const char*>(image.data()),
            image.size() * sizeof(pixel)
        );
    }

}
