#include "ppm.hpp"
#include <iostream>

int main()
{
    constexpr int width = 800;
    constexpr int height = 600;

    pixel pixels[width * height] = {
        {255,   0,   0}, // red
        {  0, 255,   0}, // green
        {  0,   0, 255}, // blue

        {255, 255,   0}, // yellow
        {  0, 255, 255}, // cyan
        {255,   0, 255}  // magenta
    };

    pixel pixels_01[3] = {
        {255,   0,   0}, // red
        {  0, 255,   0}, // green
        {  0,   0, 255} // blue
    };

    pixel background[3] = {
        {255,255,255},
        {255, 255,255},
        {255,255,255}
    };
    std::ofstream file("image.ppm", std::ios::binary);

    file << "P6\n";
    file << width << " " << height << '\n';
    file << "255\n";

    for(int h = 0; h < height; h++)
    {
        for(int w = 0; w < width; w++)
        {
            file.write(
                reinterpret_cast<const char*>(background),
                sizeof(background)
            );
        }
    }
    return 0;
}
