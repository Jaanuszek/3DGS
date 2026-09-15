#pragma once

#include "camera.hpp"
#include "glm/detail/qualifier.hpp"
#include "ppm.hpp"
#include "glm/ext/quaternion_common.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp""

#include <vector>
#include <array>
#include <memory>

namespace GS
{
    class Gaussian
    {
        public:
            Gaussian(std::shared_ptr<GS::Camera> camera, const glm::vec3& pos);

            void applyRotation(float angle, const glm::vec3& axis);

        private:
            void computeCovariance();

        private:
            glm::mat3 cov; // Covariance matrix
            glm::mat3 rotation_m;
            glm::mat3 scale_m;
            glm::mat2 cov_pix;
            glm::quat rot_q = glm::quat(1,0,0,0);
            glm::vec3 pos; // centre point
            glm::vec3 col; // color
            glm::vec2 pos_pix;
            float alpha;
    };
}
