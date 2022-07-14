cmake_minimum_required(VERSION 3.5)
project(segmentation_renderer)

add_compile_options(
  "-Wall" "-Wpedantic" "-Wextra" "-fexceptions"
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

find_package(GLM REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLFW3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(Assimp REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp
 src/renderer/renderer.cpp
 
 src/mesh_transformer/icl_mesh_transformer.cpp
 src/icl_nuim/icl_segmentation_provider.cpp

 src/mp_parser/segmentation_provider.cpp
 src/mp_parser/mp_parser.cpp
 src/mesh_transformer/mesh_transformer.cpp
 src/renderer/mp_renderer.cpp)

target_include_directories(
    ${PROJECT_NAME} PRIVATE include
    ${GLM_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${GLEW_INCLUDE_PATH}
    ${assimp_INCLUDE_DIRS})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OPENGL_LIBRARIES}
									${GLFW_LIBRARIES} ${GLEW_LIBRARY} ${assimp_LIBRARIES})
