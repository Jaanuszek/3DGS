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
#include "defines.hpp"

namespace GS
{
    struct BoundingBox
    {
        int minX,
            maxX,
            minY,
            maxY;
    };

    struct TileRange
    {
        uint32_t tile_x_min,
                tile_x_max,
                tile_y_min,
                tile_y_max;
    };

    class Gaussian
    {
        public:
            Gaussian(std::shared_ptr<GS::Camera> camera,
                const glm::vec3& pos, const glm::mat3& scale,
                const glm::vec3& col, const float opacity);

            void applyRotation(float angle, const glm::vec3& axis);

            /*
             * @param
             *      dist - the bigger dist value the bigger OBB is (larger disntace from gaussian center)
             *
             * This function creates simple AABB without taking rotation into account
             * (since it's not an OBB)
             */
            BoundingBox getBoundingBox(float dist) const;
            TileRange getRect(float dist, uint32_t grid_max_x, uint32_t grid_max_y) const;

            const glm::mat2& getInvCovPix() const { return this->inv_cov_pix; }
            const glm::vec2& getPosPix() const { return this->pos_pix; }
            const glm::vec3& getColor() const { return this->col; }
            const float& getOpacity() const { return this->opacity; }
            const float& getDepth() const { return this->depth; }
            const uint32_t& getID() const { return this->id; }
            const bool& getIsRenderable() const { return this->isRenderable; }

            void setID(const uint32_t& ID) { this->id = ID; }

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
            // if not assigned, then set to max value
            uint32_t id = std::numeric_limits<uint32_t>::max();

            bool isRenderable = false; // Is it visible in clip space?
    };
}
