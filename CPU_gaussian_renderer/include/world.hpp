#pragma once

#include "camera.hpp"
#include "gaussian.hpp"

#include <memory>
#include <vector>
#include <unordered_map>

namespace GS{

    enum class COLOR
    {
        RED,
        GREEN,
        BLUE,
        MAGENTA
    };

    const std::unordered_map<COLOR, glm::vec3> COLOR_MAP = {
       { COLOR::RED, glm::vec3(1.0f, 0.0f, 0.0f) },
       { COLOR::GREEN, glm::vec3(0.0f ,1.0f, 0.0f) },
       { COLOR::BLUE, glm::vec3(0.0f, 0.0f, 1.0f) },
       { COLOR::MAGENTA, glm::vec3(1.0f, 0.0f, 1.0f) }
    };

    class World
    {
      public:
          World(std::shared_ptr<Camera> cam);
          World(std::shared_ptr<Camera> cam,
                const std::vector<Gaussian>& g_arr);

          // template<std::ranges::input_range R>
          // World(const std::shared_ptr<Camera> cam, R&& g_arr)
          //   : gaussians(std::ranges::begin(g_arr), std::ranges::end(g_arr))
          //   {
          //       this->camera = cam;
          //   }
          //

          ~World() = default;

          World(const World&) = delete;
          World(World&&) = delete;

          World& operator=(const World&) = delete;
          World& operator=(World&&) = delete;

          void reserveGaussians(std::size_t count){ gaussians.reserve(count); }

          void addGaussian(const Gaussian& g);

          void addGaussian_wo_sorting(Gaussian g);

          void addGaussians(std::vector<Gaussian>&& g_arr);

          void sortGaussians();

          const std::vector<Gaussian>& getGaussians() const
          {
              return gaussians;
          }

          size_t getGaussiansCount()
          {
              return gaussians.size();
          }
      private:
        std::vector<Gaussian> gaussians;
        std::shared_ptr<GS::Camera> camera;
    };
}
