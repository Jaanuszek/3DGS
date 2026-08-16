#pragma once

#include "ppm.hpp"
#include "glm/glm.hpp"

#include <vector>
#include <array>

struct Gaussian
{
    // Covariance matrix should not be fixed
    // it supposed to be computed on the fly (answer is in the paper)
    // but for the simplicity, I will not compute it there XD
    glm::mat3 cov; // Covariance matrix
    glm::vec3 pos; // centre point
    glm::vec3 col; // color
    float alpha;
};

int foo();

// This funcion will return gaussian color for given pixel index
pixel getGaussianValue(uint32_t idx);

void projectGaussian(const Gaussian& g, const glm::mat4 &proj);
