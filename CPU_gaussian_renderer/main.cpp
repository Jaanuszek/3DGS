#include "glm/fwd.hpp"
#include "ppm.hpp"
#include "gaussian.hpp"
#include "camera.hpp"
#include <iostream>
#include <memory>
#include <random>
#include <chrono>
#include "world.hpp"


int main()
{
    constexpr int WIDHT = 800;
    constexpr int HEIGHT = 600;
    constexpr float FOV = 90.0f;
    constexpr uint32_t GAUSSIANS_COUNT = 100;

    std::shared_ptr<GS::Camera> camera = std::make_shared<GS::Camera>(
        WIDHT, HEIGHT, FOV
    );

    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 2.0f);
    glm::vec3 targetPos = glm::vec3(0.0f, 0.0f, 0.0f);

    camera->init(cameraPos, targetPos);

    GS::World world(camera);

    glm::mat3x3 scale_m(1.0f);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist_x(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_z(-30.0f, -5.0f);
    std::uniform_real_distribution<float> dist_color(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist_opacity(0.0f, 1.0f);

    world.reserveGaussians(GAUSSIANS_COUNT);

    auto start = std::chrono::high_resolution_clock::now();
    for(uint32_t i = 0; i < GAUSSIANS_COUNT; i++)
    {
        glm::vec3 position(
            dist_x(gen),
            dist_y(gen),
            dist_z(gen)
        );

        glm::vec3 color(
            dist_color(gen),
            dist_color(gen),
            dist_color(gen)
        );

        float opacity = dist_opacity(gen);

        world.addGaussian_wo_sorting(
            GS::Gaussian(
                camera,
                position,
                scale_m,
                color,
                opacity
            )
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Generating gaussians time: " << elapsed.count() << std::endl;

    world.sortGaussians();

    std::vector<pixel> image;
    image.resize(WIDHT * HEIGHT);

    std::ofstream file("image.ppm", std::ios::binary);

    file << "P6\n";
    file << WIDHT << " " << HEIGHT << '\n';
    file << "255\n";

    start = std::chrono::high_resolution_clock::now();
    for(int h = 0; h < HEIGHT; h++)
    {
        for(int w = 0; w < WIDHT; w++)
        {
            std::size_t idx = h * WIDHT + w;
            // pixel pix;

            std::uint8_t bg_color = 255U;

            glm::vec2 p(w + 0.5f, h + 0.5f);

            glm::vec3 c(0.0f);
            float T = 1.0f;
            for(const auto& gauss : world.getGaussians())
            {
                // If gaussian mean point is not in clip space, then ommit it
                if(!gauss.getIsRenderable())
                {
                    continue;
                }

                glm::vec2 d = p - gauss.getPosPix();

                // d^T * Sigma^-1 * d
                float q = glm::dot(d, gauss.getInvCovPix() * d);

                float density = std::exp(-0.5f * q);

                // From paper ai is given by evaluating a 2D Gaussian (`density`)
                // multiplied with learned per-pont opacity => density * opacity
                float alpha = std::min(0.99f, density * gauss.getOpacity());

                if(alpha < 1.0f / 255.0f)
                    continue;

                // Test variable to check if Transmittance is on sufficient level
                float test_T = T * (1.0f - alpha);

                if(test_T < 0.0001f)
                    break;

                // Right now test_T is T_i+1, while T is T_i
                // So firstly we calculate color, then update T
                // T -> T_(i)
                // test_T -> T_(i+1)
                c += gauss.getColor() * alpha * T;
                T = test_T;
            }

            glm::vec3 background(0.0f);
            c = glm::clamp(c + T * background, 0.0f, 1.0f);
            // pix = {
            //     static_cast<uint8_t>(c.r * 255.0f),
            //     static_cast<uint8_t>(c.g * 255.0f),
            //     static_cast<uint8_t>(c.b * 255.0f)
            // };
            image[idx] = pixel{
                static_cast<uint8_t>(c.r * 255.0f),
                static_cast<uint8_t>(c.g * 255.0f),
                static_cast<uint8_t>(c.b * 255.0f)
            };
            // file.write(
            //     reinterpret_cast<const char*>(&pix),
            //     sizeof(pixel)
            // );
        }
    }
    file.write(
        reinterpret_cast<const char*>(image.data()),
        image.size() * sizeof(pixel)
    );
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;

    std::cout << "Generating whole PPM time: " << elapsed.count() << std::endl;
    return 0;
}
