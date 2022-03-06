#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <opencv2/opencv.hpp>

#include "icl_parser.h"
#include "renderer.h"

class ICL_Renderer : public Renderer {
public:
    ICL_Renderer(string const &pathToMesh);
    ~ICL_Renderer();
    void renderTrajectory(ICL_Parser &ip, const std::string save_path = "");
};