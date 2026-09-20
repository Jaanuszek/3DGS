#include "world.hpp"
#include "gaussian.hpp"
#include <memory>
#include <ranges>

namespace GS
{
    World::World(std::shared_ptr<Camera> cam)
        : camera(std::move(cam))
    {
    }

    World::World(std::shared_ptr<Camera> cam,
          const std::vector<Gaussian>& g_arr)
        : camera(std::move(cam)), gaussians(g_arr)
    {
        sortGaussians();
    }

    void World::addGaussian(const Gaussian& g)
    {
        gaussians.push_back(g);
        sortGaussians();
    }

    void World::addGaussians(std::vector<Gaussian>&& g_arr)
    {
        for(auto& g : g_arr)
        {
            gaussians.push_back(std::move(g));
        }
        sortGaussians();
    }

    void World::addGaussian_wo_sorting(Gaussian g)
    {
        gaussians.push_back(std::move(g));
    }

    void World::sortGaussians()
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
