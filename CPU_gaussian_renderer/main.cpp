#include "glm/fwd.hpp"
#include "ppm.hpp"
#include "gaussian.hpp"
#include "camera.hpp"
#include <atomic>
#include <iostream>
#include <memory>
#include <random>
#include <chrono>
#include <thread>
#include "world.hpp"
#include "defines.hpp"
#include "renderer.hpp"

int main()
{
    constexpr uint32_t GRID_X_MAX = (CONSTANT::WIDTH + CONSTANT::TILE_X - 1) / CONSTANT::TILE_X;
    constexpr uint32_t GRID_Y_MAX = (CONSTANT::HEIGHT + CONSTANT::TILE_Y - 1) / CONSTANT::TILE_Y;
    constexpr uint32_t TILES_COUNT = GRID_X_MAX * GRID_Y_MAX;

    std::shared_ptr<GS::Camera> camera = std::make_shared<GS::Camera>(
        CONSTANT::WIDTH, CONSTANT::HEIGHT, CONSTANT::FOV
    );

    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 2.0f);
    glm::vec3 targetPos = glm::vec3(0.0f, 0.0f, 0.0f);

    camera->init(cameraPos, targetPos);

    GS::World world(camera);

    glm::mat3x3 scale_m(1.0f);

    std::random_device rd;
    // std::mt19937 gen(12345); <------ In case if I want to have seeded rand
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist_x(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_z(-30.0f, -5.0f);
    std::uniform_real_distribution<float> dist_color(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist_opacity(0.0f, 1.0f);

    world.reserveGaussians(CONSTANT::GAUSSIANS_COUNT);

    auto start = std::chrono::high_resolution_clock::now();
    for(uint32_t i = 0; i < CONSTANT::GAUSSIANS_COUNT; i++)
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

    GS::Renderer renderer(std::move(world));

    const unsigned int THREADS_COUNT = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(THREADS_COUNT);

    std::atomic<uint32_t> next_tile{0};

    start = std::chrono::high_resolution_clock::now();
    for(int t = 0; t < THREADS_COUNT; t++)
    {
        threads.emplace_back([&renderer, &next_tile]()
            {
                while(true)
                {
                    const uint32_t tile_id = next_tile.fetch_add(1, std::memory_order_relaxed);

                    if(tile_id >= TILES_COUNT)
                        break;

                    renderer.render_tile(tile_id);
                }
            }
        );
    }

    for(auto& thread : threads)
    {
        thread.join();
    }
    // std::cout << "Generating gaussians time: " << elapsed.count() << std::endl;

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;

    renderer.saveToPPM("image.ppm");
    std::cout << "Generating whole PPM time: " << elapsed.count() << std::endl;

    return 0;
}
