#pragma once

#include "glm/glm.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"

namespace GS
{
    constexpr float NEAR = 0.1f;
    constexpr float FAR = 100.0f;

    class Camera
    {
        public:
            Camera(int w, int h, float fov);

            void init(const glm::vec3& cameraPos, const glm::vec3& targetPos);

            // Multiply by projection and view matrices
            // and perform perspective division
            glm::vec2 mvp(const glm::vec4& point);

            // Point in camera view
            glm::vec4 mulView(const glm::vec4& point);

            // Point in clip space (before perspective division)
            // Input point shall be in camera view (!!)
            glm::vec4 mulProj(const glm::vec4& pointCamera);

            glm::vec2 perspectiveDivision(const glm::vec4& pointClip);

            // compute Jacobian from point in camera space
            glm::mat3x2 computeJacobian(const glm::vec4& pointCam);

            // from [-1,1] to [0, width]
            glm::vec2 NDCtoPixel(const glm::vec2 pointNDC);

            const glm::mat4& getProjMat() { return projectionMat; }
            const glm::mat4& getviewMat() { return viewMat; }
        private:


        private:
            glm::mat4 projectionMat;
            glm::mat4 viewMat;
            glm::vec3 globalUp = glm::vec3(0.0f, 1.0f, 0.0f);

            float width{},
                  height{},
                  fov{},
                  aspect_ratio{},
                  focal_x{},
                  focal_y{};
    };
}
