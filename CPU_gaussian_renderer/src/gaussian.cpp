#include "gaussian.hpp"
#include "camera.hpp"
#include <glm/ext/quaternion_common.hpp>

#include <glm/gtc/quaternion.hpp>
#include <iostream>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"

namespace GS
{
    Gaussian::Gaussian(std::shared_ptr<GS::Camera> camera,
        const glm::vec3& pos, const glm::mat3& scale,
        const glm::vec3& col, const float opacity)
    {
        this->pos = pos;
        this->scale_m = scale;
        this->rotation_m = glm::mat3_cast(glm::normalize(rot_q));
        this->opacity = std::clamp(opacity, 0.0f, 1.0f);
        this->col = col;
        this->computeCovariance();

        glm::vec4 pointCam = camera->mulView(glm::vec4(pos, 1.0f));
        glm::vec4 pointProj = camera->mulProj(pointCam);

        // It's not the greatest solution, since some gaussian point can be very near to clip space
        // and it's density may be visible from the camera point of view
        // So the improvement might be just adding gaussian distribution check in isVisible bool
        // to check wether this specific density is visible on the display.
        bool isVisible = pointProj.w > 0.0f && // to not divide by zero in perspective division
            pointProj.z >= -pointProj.w && // near
            pointProj.z <= pointProj.w && // far
            pointProj.x >= -pointProj.w && pointProj.x <= pointProj.w &&
            pointProj.y >= -pointProj.w && pointProj.y <= pointProj.w;
        if(isVisible)
        {
            this->isRenderable = true;
        }

        glm::vec2 pointNDC = camera->perspectiveDivision(pointProj);
        this->pos_pix = camera->NDCtoPixel(pointNDC);

        this->J = camera->computeJacobian(pointCam);

        glm::mat3 covView = computeCovarianceView(glm::mat3(camera->getviewMat()));
        glm::mat2 covPix = this->J * covView * glm::transpose(this->J);
        this->inv_cov_pix = glm::inverse(covPix);

        this->computeDepth(camera->getviewMat());
    }

    void Gaussian::applyRotation(float angle, const glm::vec3 &axis)
    {

    }

    BoundingBox Gaussian::getBoundingBox(float dist) const
    {
        glm::mat2 cov_m = glm::inverse(this->inv_cov_pix);
        float a = cov_m[0][0];
        float b = cov_m[0][1];
        float c = cov_m[1][0];
        float d = cov_m[1][1];

        float l = (a + d) / 2;
        float r = std::sqrt(std::pow((a - d)/2, 2) + b * b);
        float lambda1_sqrt = dist * std::sqrt(l + r);
        float lambda2_sqrt = dist * std::sqrt(l - r);

        float theta{};
        if((b < 1e-4) && (a >= d)) // ~0
        {
            theta = 0;
        }
        else if((b < 1e-4) && (a < d))
        {
            theta = std::numbers::pi / 2;
        }
        else
        {
            theta = std::atan2((lambda1_sqrt - a), b);
        }

        glm::mat2 rot_m(
          std::cos(theta), std::sin(theta), // first column
          -std::sin(theta), std::cos(theta) // second column
        );

        return BoundingBox{
            // .minX = this->pos_pix.x + rot_m * glm::vec2(lambda1_sqrt, lambda2_sqrt),
        };
    }

    void Gaussian::computeCovariance()
    {
        this->cov = rotation_m * scale_m * glm::transpose(scale_m) * glm::transpose(rotation_m);
    }


    glm::mat3 Gaussian::computeCovarianceView(const glm::mat3& view_rotation_m)
    {
        return view_rotation_m * cov * glm::transpose(view_rotation_m);
    }

    void Gaussian::computeDepth(const glm::mat4& viewMat)
    {
        glm::vec4 viewPos = viewMat * glm::vec4(pos, 1.0f);
        this->depth = -viewPos.z;
    }
}
