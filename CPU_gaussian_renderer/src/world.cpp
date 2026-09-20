#include "world.hpp"
#include "gaussian.hpp"
#include <memory>
#include <ranges>

namespace GS
{
    World::World(std::shared_ptr<Camera> cam)
        : camera(std::move(cam))
    {
        sortGuassians();
    }

    World::World(std::shared_ptr<Camera> cam,
          const std::vector<Gaussian>& g_arr)
        : camera(std::move(cam)), gaussians(g_arr)
    {
        sortGuassians();
    }

    void World::addGaussian(const Gaussian& g)
    {
        gaussians.push_back(g);
        sortGuassians();
    }

    void World::addGaussians(std::vector<Gaussian>&& g_arr)
    {
        gaussians = std::move(g_arr);
    }

    void World::sortGuassians()
    {
        if(gaussians.size() == 0)
            return;

        std::sort(gaussians.begin(), gaussians.end(),
            [](const Gaussian& g1, const Gaussian& g2)
            {
                return g1.getDepth() < g2.getDepth();
            });
    }
}
