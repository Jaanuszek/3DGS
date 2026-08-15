#include <iostream>
#include <cstdint>
#include <fstream>

int main()
{
    struct pixel{
        std::uint8_t r;
        std::uint8_t g;
        std::uint8_t b;
    };

    const int width = 3;
    const int height = 2;

    pixel pixels[width * height] = {
        {255,   0,   0}, // red
        {  0, 255,   0}, // green
        {  0,   0, 255}, // blue

        {255, 255,   0}, // yellow
        {  0, 255, 255}, // cyan
        {255,   0, 255}  // magenta
    };

    std::ofstream file("image.ppm", std::ios::binary);

    file << "P6\n";
    file << width << " " << height << '\n';
    file << "255\n";

    file.write(
        reinterpret_cast<const char*>(pixels),
        sizeof(pixels)
    );

    return 0;
}
