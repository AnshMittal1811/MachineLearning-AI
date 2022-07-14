#include "renderer.h"
#include "model.h"
// #include "util.h"

// basic file operations
#include <iostream>
#include <fstream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window, Renderer &renderer, int* imgCounter);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
bool takeScreenshot = false;
bool spacePressedAtLeastOnce = false;

// camera
Camera camera(glm::vec3(0.790932f, 1.300000f, 1.462270f)); // 1.3705f, 1.51739f, 1.44963f    0.0f, 0.0f, 3.0f      -0.3f, 0.3f, 0.3f    0.790932f, 1.300000f, 1.462270f
float lastX = DEF_WIDTH / 2.0f;
float lastY = DEF_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

Renderer::Renderer(string const &pathToMesh, int width, int height) {  
    m_buffer_width = width;
    m_buffer_height = height;

    if(init()){
        // if init fails, then the return code is != 0 which is equal to this if statement
        throw std::runtime_error("Failed to init renderer");
    }

    m_model = new Model(pathToMesh);
    m_shader = new Shader("../shader/color3D.vs", "../shader/color3D.frag");
}

Renderer::~Renderer() {
    delete &m_model;
    delete &m_shader;
    glfwTerminate();
}

int Renderer::init() {
    if(! glfwInit()){
        std::cout << "Failed to init glfw" << std::endl;
        return EXIT_FAILURE;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    // glfw window creation
    // --------------------
    m_window = glfwCreateWindow(m_buffer_width, m_buffer_height, "Segmentation_Renderer", NULL, NULL);
    if (m_window == nullptr) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(m_window);

    // To avoid: https://stackoverflow.com/questions/8302625/segmentation-fault-at-glgenvertexarrays-1-vao
    glewExperimental = GL_TRUE; 
    if (GLEW_OK != glewInit()){
        std::cout << "Failed to init glew" << std::endl;
        return EXIT_FAILURE;
    }

    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
    glfwSetCursorPosCallback(m_window, mouse_callback);
    glfwSetScrollCallback(m_window, scroll_callback);
    glfwSetKeyCallback(m_window, key_callback);

    // configure global opengl state
    glEnable(GL_DEPTH_TEST);

    m_initialized = true;
    return 0;
}

void Renderer::render(const glm::mat4& model, const glm::mat4& view, const glm::mat4& projection){
    if(! m_initialized){
        std::cout << "Cannot render before initializing the renderer" << std::endl;
        return;
    }
    GLuint framebuffername = 1;
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffername);

    glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_shader->use();

    m_shader->setMat4("projection", projection);
    m_shader->setMat4("view", view);
    m_shader->setMat4("model", model);

    m_model->draw(*m_shader);
}

void Renderer::readRGB(cv::Mat& image) {
    glBindFramebuffer(GL_FRAMEBUFFER, 4);
    image = cv::Mat(m_buffer_height, m_buffer_width, CV_8UC3);
    
    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_PACK_ALIGNMENT, (image.step & 3) ? 1 : 4);

    //set length of one complete row in destination data (doesn't need to equal img.cols)
    glPixelStorei(GL_PACK_ROW_LENGTH, image.step/image.elemSize());

    glReadPixels(0, 0, image.cols, image.rows, GL_BGR, GL_UNSIGNED_BYTE, image.data);
    cv::flip(image, image, 0);
    cv::flip(image, image, 1);
    // see: https://stackoverflow.com/questions/9097756/converting-data-from-glreadpixels-to-opencvmat/9098883
}

void Renderer::readDepth(cv::Mat& image) {
    image = cv::Mat(m_buffer_height, m_buffer_width, CV_32FC1);

    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_PACK_ALIGNMENT, (image.step & 3) ? 1 : 4);

    //set length of one complete row in destination data (doesn't need to equal img.cols)
    glPixelStorei(GL_PACK_ROW_LENGTH, image.step/image.elemSize());

    glReadPixels(0, 0, image.cols, image.rows, GL_DEPTH_COMPONENT, GL_FLOAT, image.data);

    cv::flip(image, image, 0);
    cv::flip(image, image, 1);
    // see: https://stackoverflow.com/questions/9097756/converting-data-from-glreadpixels-to-opencvmat/9098883
}

void Renderer::renderInteractive(ICL_Parser &ip){
    // tell GLFW to capture our mouse
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    // render loop
    int imgCounter = 0;
    while (!glfwWindowShouldClose(m_window))
    {

        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(m_window, *this, &imgCounter);
        
        // model/view/projection transformations
        // ------
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)m_buffer_width / (float)m_buffer_height, 0.1f, 100.0f);
        glm::mat3 intr = ip.getIntrinsics();
        // glm::mat4 projection = camera_utils::perspective(intr, ip.getWidth(), ip.getHeight(), kNearPlane, kFarPlane);

        glm::mat4 extr_scale = glm::mat4(1.0f);
        extr_scale = glm::scale(extr_scale, glm::vec3(-1, 1, 1));
        glm::mat4 view = camera.GetViewMatrix();
        view = view * extr_scale;


        glm::mat4 to_opengl_coords = glm::mat4(1.0f);
        to_opengl_coords = glm::scale(to_opengl_coords, glm::vec3(-1, -1, -1));

        // render
        // ------
        render(view, to_opengl_coords, projection);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window, Renderer &renderer, int* imgCounter)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);

    if (takeScreenshot){
        cv::Mat colorImage;
        renderer.readRGB(colorImage);

        // save matrix as file
        if (!colorImage.empty()) {
            std::stringstream image_filename;
            char image_name[30];
            sprintf(image_name, "scene_00_%04d.seg.png", *imgCounter);
            image_filename << image_name;
            cv::imwrite(image_filename.str(), colorImage);

            std::cout << "Wrote image: " << image_name << std::endl;


            // write cam matrix
            std::stringstream cam_filename;
            char cam_name[30];
            sprintf(cam_name, "scene_00_%04d.txt", *imgCounter);
            cam_filename << cam_name;

            glm::mat4 view = camera.GetViewMatrix();
            view = glm::inverse(view); // RT goes from world to view, but in ICL we save view-to-world so use this camera here as well.

            ofstream cam_file;
            cam_file.open (cam_filename.str());
            cam_file << "cam_pos\t= [" << view[3][0] << ", " << view[3][1] << ", " << view[3][2] << "]';\n";
            cam_file << "cam_dir\t= [" << view[2][0] << ", " << view[2][1] << ", " << view[2][2] << "]';\n";
            cam_file << "cam_up\t= [" << view[1][0] << ", " << view[1][1] << ", " << view[1][2] << "]';\n";
            cam_file.close();

            std::cout << "Wrote cam: " << cam_name << std::endl;

            // increment
            (*imgCounter)++;
        }

        takeScreenshot = false;
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if ( (key == GLFW_KEY_SPACE && action == GLFW_PRESS) 
       ||(key == GLFW_KEY_W && spacePressedAtLeastOnce && action == GLFW_PRESS)
       ||(key == GLFW_KEY_A && spacePressedAtLeastOnce && action == GLFW_PRESS)
       ||(key == GLFW_KEY_S && spacePressedAtLeastOnce && action == GLFW_PRESS)
       ||(key == GLFW_KEY_D && spacePressedAtLeastOnce && action == GLFW_PRESS)){
        takeScreenshot = true;
    }

    if (! spacePressedAtLeastOnce && key == GLFW_KEY_SPACE && action == GLFW_PRESS){
        spacePressedAtLeastOnce = true;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}