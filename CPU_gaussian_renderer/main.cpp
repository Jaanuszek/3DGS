#include "glm/ext/quaternion_common.hpp"
#include "glm/fwd.hpp"
#include "glm/matrix.hpp"
#include "ppm.hpp"
#include "gaussian.hpp"
#include "camera.hpp"
#include <iostream>
#include <memory>

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

    // create covariance matrix from scale matrix and rotation matrix
    // In paper rotation matrix is taken from quaternion
    glm::vec3 euler(1.0f, 1.0f, 1.0f);

    // quaternion ha to be normalized (pure rotation quaternion has to have length of 1 ||q|| = 1)
    glm::quat norm_rotation_q = glm::normalize(glm::quat(euler));

    // from quaterinion to 3x3 matrix
    glm::mat3x3 rotation_m = glm::mat3_cast(norm_rotation_q);

    // rotation is stored in vector, but it is converted to matrix
    glm::mat3x3 scale_m{};
    scale_m[0][0] = 1.0f;
    scale_m[1][1] = 1.0f;
    scale_m[2][2] = 1.0f;

    glm::mat3x3 covariance_m = rotation_m * scale_m * glm::transpose(scale_m) * glm::transpose(rotation_m);

    glm::vec3 pointL(0.0f, 0.0f, -10.0f);

    glm::vec4 pointCam = camera->mulView(glm::vec4(pointL, 1.0f));
    glm::vec4 pointProj = camera->mulProj(pointCam);
    glm::vec2 pointNDC = camera->perspectiveDivision(pointProj);
    glm::vec2 pointScreen = camera->NDCtoPixel(pointNDC);

    glm::mat3x2 J = camera->computeJacobian(pointCam);
    glm::mat3 rotMatrix(camera->getviewMat());

    // manually created these 3 steps, just because final result will be glm::mat2
    // but covarianceView mat3. It's propably not necessery
    glm::mat3 covarianceView = rotMatrix * covariance_m * glm::transpose(rotMatrix);

    glm::mat2 covariancePixel = J * covarianceView * glm::transpose(J);

    glm::mat2 invCovariancePixel = glm::inverse(covariancePixel);

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

            glm::vec2 d = p - pointScreen;

            // d^T * Sigma^-1 * d
            float q = glm::dot(d, invCovariancePixel * d);

            float density = std::exp(-0.5f * q);

            int c_int = static_cast<int>(255.0f * density);
            std::uint8_t c_uint = static_cast<std::uint8_t>(c_int);

            if(h == static_cast<int>(pointScreen.y) && w == static_cast<int>(pointScreen.x))
            {
                bg_color = 0U;
            }
            pix = {c_uint,c_uint,c_uint};
            file.write(
                reinterpret_cast<const char*>(&pix),
                sizeof(pixel)
            );
        }
    }
    return 0;
}
