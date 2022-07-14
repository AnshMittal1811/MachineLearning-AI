#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <opencv2/opencv.hpp>

#include "mp_parser.h"
#include "renderer.h"

class MP_Renderer : public Renderer {

public:
    MP_Renderer(string const &pathToMesh, MP_Parser const &mp_parser, int region_index);
    ~MP_Renderer();
    void renderImages(const std::string save_path = "");

private:
    MP_Parser mp_parser;
    int region_index;
};