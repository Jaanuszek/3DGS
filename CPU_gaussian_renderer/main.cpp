#include "glm/fwd.hpp"
#include "ppm.hpp"
#include "gaussian.hpp"
#include "camera.hpp"
#include <iostream>
#include <memory>
#include "world.hpp"


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

    std::unique_ptr<GS::World> world;

    glm::mat3x3 scale_m{};
    scale_m[0][0] = 1.0f;
    scale_m[1][1] = 1.0f;
    scale_m[2][2] = 1.0f;

    glm::vec3 pointL(0.0f, 0.0f, -15.0f);
    glm::vec3 point2(1.0f, 1.0f, -5.0f);
    glm::vec3 point3(-50.0f, -5.0f, -20.0f);
    glm::vec3 point4(-5.0f, 5.0f, -30.0f);

    GS::Gaussian gs1(camera, pointL, scale_m, glm::vec3(1.0f, 0.0f, 0.0f), 1.5f);
    GS::Gaussian gs2(camera, point2, scale_m, glm::vec3(0.0f, 1.0f, 0.0f), 0.5f);
    GS::Gaussian gs3(camera, point3, scale_m, glm::vec3(0.0f, 0.0f, 1.0f), 1.0f);
    GS::Gaussian gs4(camera, point4, scale_m, glm::vec3(1.0f, 0.0f, 1.0f), 0.8f);

    std::vector<GS::Gaussian> gaussians= {
      gs1,
      gs2,
      gs3,
      gs4
    };

    world = std::make_unique<GS::World>(camera, gaussians);

    world->addGaussian(GS::Gaussian(
        camera, glm::vec3(5.0f, -5.0f, -5.0f), scale_m, glm::vec3(0.0f, 1.f, 1.f), 1.0f
    ));

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

            glm::vec3 c(0.0f);
            float T = 1.0f;
            for(const auto& gauss : world->getGaussians())
            {
                // If gaussian mean point is not in clip space, then ommit it
                if(!gauss.getIsRenderable())
                {
                    continue;
                }

                glm::vec2 gaussPoint = gauss.getPosPix();
                glm::vec2 d = p - gaussPoint;

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
            pix = {
                static_cast<uint8_t>(c.r * 255.0f),
                static_cast<uint8_t>(c.g * 255.0f),
                static_cast<uint8_t>(c.b * 255.0f)
            };
            file.write(
                reinterpret_cast<const char*>(&pix),
                sizeof(pixel)
            );
        }
    }
    return 0;
}
