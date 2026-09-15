#include "camera.hpp"

namespace GS
{
    Camera::Camera(int w, int h, float fov)
    {
        this->width = w;
        this->height = h;
        this->fov = fov;

        aspect_ratio = static_cast<float>(width) / static_cast<float>(height);

        projectionMat = glm::perspective(
            glm::radians(fov),
            aspect_ratio,
            NEAR,
            FAR
        );

        // glm::perspecitve creates projection in NDC [-1,1], but we need it in pixel values
        // X and Y diagonal of this matrix are related with focal x and focal y
        focal_x = projectionMat[0][0] * width * 0.5f;
        focal_y = projectionMat[1][1] * height * 0.5f;
    }

    void Camera::init(const glm::vec3& cameraPos, const glm::vec3& targetPos)
    {
        viewMat = glm::lookAt(cameraPos, targetPos, globalUp);
    }

    glm::vec2 Camera::mvp(const glm::vec4& point)
    {
        glm::vec4 pointClip = this->projectionMat * this->viewMat * point;
        return this->perspectiveDivision(pointClip);
    }

    glm::vec4 Camera::mulView(const glm::vec4& point)
    {
        return this->viewMat * point;
    }

    glm::vec4 Camera::mulProj(const glm::vec4& pointView)
    {
        return this->projectionMat * pointView;
    }

    glm::vec2 Camera::perspectiveDivision(const glm::vec4& pointClip)
    {
        return glm::vec2(pointClip.x/pointClip.w, pointClip.y/pointClip.w);
    }

    glm::mat3x2 Camera::computeJacobian(const glm::vec4& pointCam)
    {
        const float S = 1.0f / (pointCam.z * pointCam.z);
        return glm::mat3x2(
            focal_x / pointCam.z, 0.0f,
            0.0f, focal_y /pointCam.z,
            -(focal_x * pointCam.x) * S, -(focal_y * pointCam.y) * S
        );
    }

    glm::vec2 Camera::NDCtoPixel(const glm::vec2 pointNDC)
    {
        return glm::vec2(
          (pointNDC.x + 1.0f) * 0.5f  * width,
          (1.0f - pointNDC.y) * 0.5f * height
        );
    }
}
