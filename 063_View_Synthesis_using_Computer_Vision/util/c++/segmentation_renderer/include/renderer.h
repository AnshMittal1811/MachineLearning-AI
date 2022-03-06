#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <opencv2/opencv.hpp>

#include "model.h"
#include "camera.h"

#include "icl_parser.h"

const unsigned int DEF_WIDTH = 1280;
const unsigned int DEF_HEIGHT = 1024;

constexpr float kNearPlane{0.1f};
constexpr float kFarPlane{10.0f};

class Renderer {
public:
    Renderer(string const &pathToMesh, int width, int height);
    ~Renderer();
    void renderInteractive(ICL_Parser &ip);
    int init();

    void readRGB(cv::Mat& image);
    void readDepth(cv::Mat& image);

    Model* m_model = nullptr;
protected:

    int m_buffer_width = DEF_WIDTH;
    int m_buffer_height = DEF_HEIGHT;
    bool m_initialized = false;

    GLFWwindow* m_window = nullptr;
    Shader* m_shader = nullptr;

    void render(const glm::mat4& model, const glm::mat4& view, const glm::mat4& projection);
};