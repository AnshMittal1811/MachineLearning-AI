cmake_minimum_required(VERSION 3.5)
project(ply_transformer)

add_compile_options(
  "-Wall" "-Wpedantic" "-Wextra" "-fexceptions"
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

add_executable(${PROJECT_NAME} src/main.cpp)

target_include_directories(
    ${PROJECT_NAME} PRIVATE include
)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OPENGL_LIBRARIES}
									${GLFW_LIBRARIES} ${GLEW_LIBRARY} ${assimp_LIBRARIES})
