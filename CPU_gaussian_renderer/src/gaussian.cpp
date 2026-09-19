#include "gaussian.hpp"
#include "camera.hpp"
#include <glm/ext/quaternion_common.hpp>

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
