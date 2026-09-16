#include "gaussian.hpp"
#include "camera.hpp"
#include <glm/ext/quaternion_common.hpp>

namespace GS
{
    Gaussian::Gaussian(std::shared_ptr<GS::Camera> camera,
        const glm::vec3& pos, const glm::mat3& scale)
    {
        this->pos = pos;
        this->scale_m = scale;
        this->rotation_m = glm::mat3_cast(glm::normalize(rot_q));
        this->computeCovariance();

        glm::vec4 pointCam = camera->mulView(glm::vec4(pos, 1.0f));
        glm::vec4 pointProj = camera->mulProj(pointCam);
        glm::vec2 pointNDC = camera->perspectiveDivision(pointProj);
        this->pos_pix = camera->NDCtoPixel(pointNDC);

        this->J = camera->computeJacobian(pointCam);

        // cast from mat4 to mat3, since rotation is included in upper-left 3x3 sub-matrix
        this-> rotation_m = glm::mat3(camera->getviewMat());

        glm::mat3 covView = computeCovarianceView();
        glm::mat2 covPix = this->J * covView * glm::transpose(this->J);
        this->inv_cov_pix = glm::inverse(covPix);

    }

    void Gaussian::applyRotation(float angle, const glm::vec3 &axis)
    {

    }

    void Gaussian::computeCovariance()
    {
        this->cov = rotation_m * scale_m * glm::transpose(scale_m) * glm::transpose(rotation_m);
    }


    glm::mat3 Gaussian::computeCovarianceView()
    {
        return rotation_m * cov * glm::transpose(rotation_m);
    }
}
