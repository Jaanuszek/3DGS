#pragma once

#include "gaussian.hpp"

#include <vector>

namespace GS{
    class world
    {
      public:
          world();

          void addGaussian(const Gaussian& g);

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
    };
}
