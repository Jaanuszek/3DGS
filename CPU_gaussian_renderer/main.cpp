#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/quaternion_common.hpp"
#include "glm/fwd.hpp"
#include "glm/matrix.hpp"
#include "ppm.hpp"
#include "gaussian.hpp"
#include "projection.hpp"
#include <iostream>

#include "glm/gtc/quaternion.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"

int main()
{
    constexpr int WIDHT = 800;
    constexpr int HEIGHT = 600;
    constexpr float FOV = 90.0f;
    constexpr float ASPECT_RATIO = static_cast<float>(WIDHT) / static_cast<float>(HEIGHT);
    constexpr float NEAR = 0.1f;
    constexpr float FAR = 100.0f;

    // create covariance matrix from scale matrix and rotation matrix
    // In paper rotation matrix is taken from quaternion
    glm::vec3 euler(1.0f, 1.0f, 1.0f);

    // quaternion ha to be normalized (pure rotation quaternion has to have length of 1 ||q|| = 1)
    glm::quat norm_rotation_q = glm::normalize(glm::quat(euler));

    // from quaterinion to 3x3 matrix
    glm::mat3x3 rotation_m = glm::mat3_cast(norm_rotation_q);

    // rotation is stored in vector, but it is converted to matrix
    glm::mat3x3 scale_m{};
    scale_m[0][0] = 5.0f;
    scale_m[1][1] = 5.0f;
    scale_m[2][2] = 3.0f;

    glm::mat3x3 covariance_m = rotation_m * scale_m * glm::transpose(scale_m) * glm::transpose(rotation_m);

    glm::mat4 projectionMat = glm::perspective(
        glm::radians(FOV),
        ASPECT_RATIO,
        NEAR,
        FAR
    );

    // glm::perspecitve creates projection in NDC [-1,1], but we need it in pixel values
    // X and Y diagonal of this matrix are related with focal x and focal y
    // fx = P00 * W/2
    // fy = P11 * H/2
    float fx = projectionMat[0][0] * WIDHT * 0.5f;
    float fy = projectionMat[1][1] * HEIGHT * 0.5f;

    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 2.0f);
    glm::vec3 targetPos = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 globalUp = glm::vec3(0.0f, 1.0f, 0.0f);

    auto myLookAt = My::lookAt(cameraPos, targetPos, globalUp);

    glm::vec3 pointL(1.0f, -2.0f, -10.0f);

    glm::vec4 pointCam = myLookAt * glm::vec4(pointL, 1.0f);

    glm::vec4 pointProj = projectionMat * pointCam;

    glm::vec2 pointNDC(pointProj.x/pointProj.w, pointProj.y/pointProj.w);

    glm::vec2 pointScreen(
        (pointNDC.x + 1.0f) / 2.0f * WIDHT,
        (1.0f - pointNDC.y) / 2.0f * HEIGHT
    );

    const float S = 1.0 / (pointCam.z * pointCam.z);

    glm::mat3x2 J(
        fx / pointCam.z, 0.0f,
        0.0f, fy /pointCam.z,
        -(fx * pointCam.x) * S, -(fy * pointCam.y) * S
    );

    std::cout << glm::to_string(J) << std::endl;

    glm::mat3 rotMatrix(myLookAt);

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
            // [-1.0,1.0]
            float x = 2.0f * (w + 0.5f) / static_cast<float>(WIDHT - 1) - 1.0f;
            float y = 2.0f * (h + 0.5f) / static_cast<float>(HEIGHT - 1) - 1.0f;
            float z = 0;

            float invY = -y; // since screen is rendered from upper left corner to lower right corner

            pixel pix;

            std::uint8_t bg_color = 255U;

            int dx = w - static_cast<int>(pointScreen.x);
            int dy = h - static_cast<int>(pointScreen.y);

            if(dx * dx + dy * dy <= radiusScreen * radiusScreen)
            {
                pix = {255U, 0U, 0U};
            }
            else
            {
                pix = {255U, 255U, 255U};
            }

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
