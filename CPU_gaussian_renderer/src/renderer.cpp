#include "renderer.hpp"
#include "defines.hpp"
#include "ppm.hpp"
#include "world.hpp"

namespace GS
{
    Renderer::Renderer(World&& world) : world(std::move(world))
    {
        constexpr uint32_t PIXEL_COUNT = CONSTANT::WIDTH * CONSTANT::HEIGHT;

        this->GRID_X_MAX = (CONSTANT::WIDTH + CONSTANT::TILE_X - 1) / CONSTANT::TILE_X;
        this->GRID_Y_MAX = (CONSTANT::HEIGHT + CONSTANT::TILE_Y - 1) / CONSTANT::TILE_Y;

        image.resize(PIXEL_COUNT);

        Tiles = std::vector<std::vector<uint32_t>>(this->GRID_X_MAX * this->GRID_Y_MAX);

        this->gaussToTile();
        this->sortTiles();
    }

    void Renderer::render_tile(uint32_t tile_id)
    {
        const uint32_t tile_x = tile_id % GRID_X_MAX;
        const uint32_t tile_y = tile_id / GRID_X_MAX;
        const uint32_t minX = tile_x * CONSTANT::TILE_X;
        const uint32_t maxX = std::min(minX + CONSTANT::TILE_X, CONSTANT::WIDTH);
        const uint32_t minY = tile_y * CONSTANT::TILE_Y;
        const uint32_t maxY = std::min(minY + CONSTANT::TILE_Y, CONSTANT::HEIGHT);

        const std::vector<uint32_t>& tile = Tiles[tile_id];
        const auto& gaussians = world.getGaussians();
        // tile_id.fetch_add(1, std::memory_order_relaxed);
        tile_id++;
        // std::cout << "minx: " << minX << " maxX: " << maxX << std::endl;
        // std::cout << "miny: " << minY << " maxY: " << maxY << std::endl;
        for(int h = minY; h < maxY; h++)
        {
            for(int w = minX; w < maxX; w++)
            {
                std::size_t idx = h * CONSTANT::WIDTH + w;
                glm::vec2 p(w + 0.5f, h + 0.5f);
                glm::vec3 C(0.0f);
                float T = 1.0f;
                for(const auto& g_id : tile)
                {
                    const Gaussian& gauss = gaussians[g_id];

                    const glm::vec2 d = p - gauss.getPosPix();

                    // d^T * Sigma^-1 * d
                    float q = glm::dot(d, gauss.getInvCovPix() * d);

                    float density = std::exp(-0.5f * q);

                    // From paper ai is given by evaluating a 2D Gaussian (`density`)
                    // multiplied with learned per-pont opacity => density * opacity
                    float alpha = std::min(0.99f, density * gauss.getOpacity());

                    if(alpha < 1.0f / 255.0f)
                        continue;

                    // Test variable to check if Transmittance is on sufficient level
                    // float test_T = T_ref * (1.0f - alpha);
                    C += gauss.getColor() * (T * alpha);
                    T *= (1.0f - alpha);
                    if(T < 0.0001f)
                        break;
                }
                glm::vec3 background(0.0f);
                C = glm::clamp(C, 0.0f, 1.0f);
                image[idx] += pixel{
                    static_cast<uint8_t>(C.r * 255.0f),
                    static_cast<uint8_t>(C.g * 255.0f),
                    static_cast<uint8_t>(C.b * 255.0f)
                };
            }
        }
    }

    void Renderer::saveToPPM(const std::string& out_path)
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

    void Renderer::gaussToTile()
    {
        for(const auto& gauss : this->world.getGaussians())
        {
            GS::TileRange tile_range = gauss.getRect(CONSTANT::AABB_DIST, GRID_X_MAX, GRID_Y_MAX);

            const uint32_t minX = std::max((uint32_t)0, tile_range.tile_x_min);
            const uint32_t maxX = std::min(GRID_X_MAX, tile_range.tile_x_max);
            const uint32_t minY = std::max((uint32_t)0, tile_range.tile_y_min);
            const uint32_t maxY = std::min(GRID_Y_MAX, tile_range.tile_y_max);

            const uint32_t g_id = gauss.getID();
            for(int y = minY; y < maxY; y++)
            {
                for(int x = minX; x < maxX; x++)
                {
                    const uint32_t tileID = y * GRID_X_MAX + x;
                    Tiles[tileID].push_back(g_id);
                }
            }
        }
    }

    void Renderer::sortTiles()
    {
        for(auto& tile : this->Tiles)
        {
            std::sort(tile.begin(), tile.end(),
                [this](const uint32_t& g_id1, const uint32_t& g_id2)
                {
                    return world.getGaussianDepth(g_id1) < world.getGaussianDepth(g_id2);
                }
            );
        }
    }
}
