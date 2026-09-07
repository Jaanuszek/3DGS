#include "projection.hpp"


glm::mat4 My::lookAt(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up)
{
    // glm::mat4 refMat = glm::lookAt(eye, center, up);
    // for RH f has to be negative value. It can be obtained by simply swapping
    // eye and center in below line or by adding appropriate sing in result matrix
    glm::vec3 f(glm::normalize(center - eye)); // Z axis
    glm::vec3 r(glm::normalize(glm::cross(f, up))); // X axis
    glm::vec3 u(glm::cross(r,f));

    glm::mat4 res(1); // this constructor initialize diagonal values of this matrix to 1

    res[0][0] = r.x;
    res[1][0] = r.y;
    res[2][0] = r.z;
    res[0][1] = u.x;
    res[1][1] = u.y;
    res[2][1] = u.z;
    res[0][2] = -f.x;
    res[1][2] = -f.y;
    res[2][2] = -f.z;
    res[3][0] = -glm::dot(r, eye);
    res[3][1] = -glm::dot(u, eye);
    res[3][2] = glm::dot(f, eye); // Z is positive!!

    return res;
}
