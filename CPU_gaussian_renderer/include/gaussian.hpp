#pragma once

#include "camera.hpp"
#include "ppm.hpp"
#include "glm/ext/quaternion_common.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"

#include <vector>
#include <array>
#include <memory>
#include <algorithm>

namespace GS
{
    class Gaussian
    {
        public:
            Gaussian(std::shared_ptr<GS::Camera> camera,
                const glm::vec3& pos, const glm::mat3& scale,
                const glm::vec3& col, const float opacity);

            void applyRotation(float angle, const glm::vec3& axis);

            const glm::mat2& getInvCovPix() const { return this->inv_cov_pix; }
            const glm::vec2& getPosPix() const { return this->pos_pix; }
            const glm::vec3& getColor() const { return this->col; }
            const float& getOpacity() const { return this->opacity; }
            const float& getDepth() const { return this->depth; }

        private:
            void computeCovariance();
            glm::mat3 computeCovarianceView(const glm::mat3& view_roatation_m);
            void computeDepth(const glm::mat4& viewMat);
        private:
            glm::mat3 cov; // Covariance matrix
            glm::mat3 rotation_m;
            glm::mat3 scale_m;
            glm::mat3x2 J;
            glm::mat2 inv_cov_pix; // covariance in pixel (after jacobian multiplication)
            glm::quat rot_q = glm::quat(1,0,0,0);
            glm::vec3 pos; // centre point
            glm::vec3 col; // color
            glm::vec2 pos_pix;
            float opacity;
            float depth;
    };
}
