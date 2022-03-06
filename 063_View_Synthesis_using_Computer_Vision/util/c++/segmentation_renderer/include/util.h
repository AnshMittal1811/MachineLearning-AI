#pragma once

#include <vector>

#include <glm/glm.hpp>

#include <iostream>

namespace camera_utils {

glm::mat4 perspective(glm::mat3& intrinsics, int width, int height, double n, double f) {
    assert(f > n);

    // std::cout << "fx: " << intrinsics[0][0] << std::endl;
    // std::cout << "fy: " << intrinsics[1][1] << std::endl;
    // std::cout << "cx: " << intrinsics[0][2] << std::endl;
    // std::cout << "cy: " << intrinsics[1][2] << std::endl;

    glm::mat4 res(2 * intrinsics[0][0] / width, 0, -(2*(intrinsics[0][2] / width) -1), 0,
                  0, 2 * intrinsics[1][1] / height, -(2*(intrinsics[1][2] / height) -1), 0,
                  0, 0,  -(f+n)/(f-n), -2*f*n/(f-n),
                  0, 0, -1, 0);

    // std::cout << glm::to_string(res_old) << std::endl;

    // glm::mat4 res(intrinsics[0][0] / intrinsics[0][2], 0, 0, 0,
    //               0, intrinsics[1][1] / intrinsics[1][2], 0, 0,
    //               0, 0,  -(f+n)/(f-n), -2*f*n/(f-n),
    //               0, 0, -1, 0);

    // std::cout << glm::to_string(res) << std::endl;

    res = glm::transpose(res); // I have written it in row major but glm uses column major... I hate this stuff.

    return res;
}

} // namespace camera_utils
