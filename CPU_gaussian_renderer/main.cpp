#include "glm/ext/quaternion_common.hpp"
#include "glm/fwd.hpp"
#include "glm/matrix.hpp"
#include "ppm.hpp"
#include "gaussian.hpp"
#include "camera.hpp"
#include <iostream>
#include <memory>
#include <numbers>

#include "glm/gtc/quaternion.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"

int main()
{
    constexpr int WIDHT = 800;
    constexpr int HEIGHT = 600;
    constexpr float FOV = 90.0f;

    std::shared_ptr<GS::Camera> camera = std::make_shared<GS::Camera>(
        WIDHT, HEIGHT, FOV
    );

    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 2.0f);
    glm::vec3 targetPos = glm::vec3(0.0f, 0.0f, 0.0f);

    camera->init(cameraPos, targetPos);

    glm::mat3x3 scale_m{};
    scale_m[0][0] = 1.0f;
    scale_m[1][1] = 1.0f;
    scale_m[2][2] = 1.0f;

    glm::vec3 pointL(0.0f, 0.0f, -10.0f);
    glm::vec3 point2(5.0f, 5.0f, -5.0f);
    glm::vec3 point3(-5.0f, -5.0f, -10.0f);
    glm::vec3 point4(-5.0f, 5.0f, -20.0f);

    GS::Gaussian gs1(camera, pointL, scale_m, glm::vec3(1.0f, 0.0f, 0.0f), 0.5f);
    GS::Gaussian gs2(camera, pointL, scale_m, glm::vec3(0.0f, 1.0f, 0.0f), 1.0f);
    GS::Gaussian gs3(camera, point3, scale_m, glm::vec3(0.0f, 0.0f, 1.0f), 1.0f);
    GS::Gaussian gs4(camera, point4, scale_m, glm::vec3(1.0f, 0.0f, 1.0f), 0.8f);

    std::array<GS::Gaussian, 4> gaussians= {
      gs1,
      gs2,
      gs3,
      gs4
    };

    std::vector<float> framebuffer(WIDHT*HEIGHT, 0.0f);

    std::ofstream file("image.ppm", std::ios::binary);

    file << "P6\n";
    file << WIDHT << " " << HEIGHT << '\n';
    file << "255\n";

    for(int h = 0; h < HEIGHT; h++)
    {
        for(int w = 0; w < WIDHT; w++)
        {
            pixel pix;

            std::uint8_t bg_color = 255U;

            glm::vec2 p(w + 0.5f, h + 0.5f);

            glm::vec3 pix_col(0.0f);
            float T = 1.0f;
            for(const auto& gauss : gaussians)
            {
                glm::vec2 gaussPoint = gauss.getPosPix();
                glm::vec2 d = p - gaussPoint;

                // d^T * Sigma^-1 * d
                float q = glm::dot(d, gauss.getInvCovPix() * d);

                float density = std::exp(-0.5f * q) * gauss.getAlpha();

                pix_col += T * density * gauss.getColor();
                T *= (1.0f - density);
                if(T < 0.001f)
                {
                    break;
                }
            }

            glm::vec3 background(1.0f);
            pix_col = glm::clamp(pix_col, 0.0f, 1.0f) + T * background;
            pix = {
                static_cast<uint8_t>(pix_col.r * 255.0f),
                static_cast<uint8_t>(pix_col.g * 255.0f),
                static_cast<uint8_t>(pix_col.b * 255.0f)
            };
            file.write(
                reinterpret_cast<const char*>(&pix),
                sizeof(pixel)
            );
        }
    }
    return 0;
}
