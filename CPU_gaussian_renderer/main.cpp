#include "ppm.hpp"
#include "gaussian.hpp"
#include <iostream>

int main()
{
    constexpr int width = 1920;
    constexpr int height = 1080;


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

    glm::vec2 mean(0.0f, 0.0f);
    glm::mat2 covariance(
        0.10f,0.08f,
        0.08f, 0.20f
    );

    glm::mat2 invCov = glm::inverse(covariance);

    std::ofstream file("image.ppm", std::ios::binary);

    file << "P6\n";
    file << width << " " << height << '\n';
    file << "255\n";

    for(int h = 0; h < height; h++)
    {
        for(int w = 0; w < width; w++)
        {
            // [-1.0,1.0]
            float x = 2.0f * (w + 0.5f) / static_cast<float>(width - 1) - 1.0f;
            float y = 2.0f * (h + 0.5f) / static_cast<float>(height - 1) - 1.0f;

            glm::vec2 p(x,y);

            glm::vec2 d = p - mean;

            // d^T * Sigma^-1 * d
            float q = glm::dot(d, invCov * d);

            float density = std::exp(-0.5f * q);

            int c_int = static_cast<int>(255.0f * density);
            std::uint8_t c_uint = static_cast<std::uint8_t>(c_int);

            pixel pix = {c_uint,c_uint,c_uint};

            file.write(
                reinterpret_cast<const char*>(&pix),
                sizeof(pixel)
            );
        }
    }
    return 0;
}
