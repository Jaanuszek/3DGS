#include "glm/ext/matrix_clip_space.hpp"
#include "glm/fwd.hpp"
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

    My::vec<float,3> vec3 = {1.0f, 2.0f, 3.0f};

    std::array<std::array<float,3>,2> abcd = {
      {
          {1.0f,2.0f,3.0f},
          {3.0f, 2.0f, 1.0f}
      }
    };

    My::mat<float,4,3> mat4x3 = {
        {
            1.0f, 1.0f, 3.0f,
            3.0f, 2.0f, 1.0f,
            7.0f, 6.0f, 5.0f,
            8.0f, 9.0f, 10.0f
        }
    };

    pixel pixels[WIDHT * HEIGHT] = {
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

    std::cout << glm::to_string(covariance_m) << std::endl;

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

    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec3 targetPos = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 globalUp = glm::vec3(0.0f, 1.0f, 0.0f);

    auto myLookAt = My::lookAt(cameraPos, targetPos, globalUp);
    // auto lookAt = glm::lookAt(cameraPos, targetPos, globalUp);
    std::cout << glm::to_string(myLookAt) << std::endl;
    // glm::mat3 rotLookAt =

    glm::vec3 pointL(0.0f, 0.0f, -20.0f);

    glm::vec4 pointCam = myLookAt * glm::vec4(pointL, 1.0f);

    glm::vec4 pointProj = projectionMat * pointCam;

    glm::vec2 pointNDC(pointProj.x/pointProj.w, pointProj.y/pointProj.w);

    glm::vec2 pointScreen(
        (pointNDC.x + 1.0f) / 2.0f * WIDHT,
        (1.0f - pointNDC.y) / 2.0f * HEIGHT
    );

    std::cout << "CAM: "
              << pointCam.x << " "
              << pointCam.y << " "
              << pointCam.z << " "
              << pointCam.w << "\n";

    std::cout << "CLIP: "
              << pointProj.x << " "
              << pointProj.y << " "
              << pointProj.z << " "
              << pointProj.w << "\n";

    std::cout << "NDC: "
              << pointProj.x / pointProj.w << " "
              << pointProj.y / pointProj.w << "\n";

    constexpr int RADIUS = 3;
    float radiusScreen = RADIUS * fy / -pointCam.z;
    std::cout << "X: " << pointScreen.x << " Y: " << pointScreen.y << std::endl;

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
            // const float S = 1.0 / (z * z);
            // // glm::mat2x3 2 columns 3 rows
            // glm::mat2x3 J = glm::mat2x3(
            //     fx / z, 0, -(fx * x) * S,
            //     0, fy / z, -(fy * y) * S
            // );

            // // std::cout << glm::to_string(J) << std::endl;

            // glm::vec2 p(x,y);

            // glm::vec2 d = p - mean;

            // // d^T * Sigma^-1 * d
            // float q = glm::dot(d, invCov * d);

            // float density = std::exp(-0.5f * q);

            // int c_int = static_cast<int>(255.0f * density);
            // std::uint8_t c_uint = static_cast<std::uint8_t>(c_int);

            // if(h == static_cast<int>(pointScreen.y) && w == static_cast<int>(pointScreen.x))
            // {
            //     bg_color = 0U;
            // }
            // pixel pix = {c_uint,c_uint,c_uint};
            // pixel pix = {bg_color, bg_color, bg_color};
            file.write(
                reinterpret_cast<const char*>(&pix),
                sizeof(pixel)
            );
        }
    }
    return 0;
}
