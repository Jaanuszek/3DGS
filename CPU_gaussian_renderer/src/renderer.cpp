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

        colorBuffer = std::vector<glm::vec3>(PIXEL_COUNT, glm::vec3(0.0f));
        transmittanceBuffer = std::vector<float>(PIXEL_COUNT, 1.0f);
        Tiles = std::vector<std::vector<uint32_t>>(this->GRID_X_MAX * this->GRID_Y_MAX);

        this->gaussToTile();
        this->sortTiles();
    }

    void Renderer::render_tile()
    {
        uint32_t col = std::floor(tile_id / this->GRID_X_MAX);
        uint32_t minX = tile_id * CONSTANT::TILE_X % (CONSTANT::WIDTH);
        uint32_t maxX = (tile_id * CONSTANT::TILE_X + CONSTANT::TILE_X) % (CONSTANT::WIDTH);
        uint32_t minY = col * CONSTANT::TILE_Y;
        uint32_t maxY = (minY + CONSTANT::TILE_Y) < CONSTANT::HEIGHT ? minY + CONSTANT::TILE_Y : CONSTANT::HEIGHT - 1;
        this->tile_id.fetch_add(1, std::memory_order_relaxed);
        // std::cout << "minx: " << minX << " maxX: " << maxX << std::endl;
        // std::cout << "miny: " << minY << " maxY: " << maxY << std::endl;
        for(int h = minY; h < maxY; h++)
        {
            for(int w = minX; w < maxX; w++)
            {
                for(const auto& g_id : Tiles[tile_id])
                {
                    GS::Gaussian gauss = this->world.getGaussian(g_id);
                    std::size_t idx = h * CONSTANT::WIDTH + w;
                    std::uint8_t bg_color = 255U;

                    glm::vec2 p(w + 0.5f, h + 0.5f);

                    glm::vec3 temp_c(0.0f);
                    glm::vec3 &c_ref = colorBuffer[idx];
                    float &T_ref = transmittanceBuffer[idx];
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
                    float test_T = T_ref * (1.0f - alpha);

                    if(test_T < 0.0001f)
                        continue;

                    // Right now test_T is T_i+1, while T is T_i
                    // So firstly we calculate color, then update T
                    // T -> T_(i)
                    // test_T -> T_(i+1)
                    temp_c += gauss.getColor() * alpha * T_ref;
                    T_ref = test_T;

                    glm::vec3 background(0.0f);
                    c_ref = glm::clamp(temp_c + T_ref * background, 0.0f, 1.0f);
                    image[idx] += pixel{
                        static_cast<uint8_t>(c_ref.r * 255.0f),
                        static_cast<uint8_t>(c_ref.g * 255.0f),
                        static_cast<uint8_t>(c_ref.b * 255.0f)
                    };
                }
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
