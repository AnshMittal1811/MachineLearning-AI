#pragma once

#include "mesh.h"

#include "json.hpp"
// for convenience
using json = nlohmann::json;

#include <map>
#include <glm/glm.hpp>

class Segmentation_Provider {

    public:
        Segmentation_Provider(string const &semseg_path, string const &vseg_path, string const &fseg_path, Mesh &mesh);
        ~Segmentation_Provider();
        void change_colors(Mesh &mesh);
        int getObjectId(int vertexIndex);

        // static const std::map<std::string, std::vector<int>> label_to_color;

    private:
        int n_objects;
        int n_vertices;
        json vseg;
        json semseg;
        json fseg;
        
        std::vector<int> vertex_to_segment;
        
        std::vector<int> face_to_segment;
        std::map<int, int> vertex_to_face;
        
        std::map<int, int> segment_to_object_id;
        std::map<int, glm::vec3> object_id_to_color;
        
        std::map<int, std::string> segment_to_class;
        std::map<std::string, glm::vec3> class_to_color;

        void random_rgb(float rgb[]){
            int i;
            for(i=0;i<3;i++){
                // this produces a random float number between 0 and 1 inclusively
                rgb[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
        }
};

// MIGHT NEED TO PARSE THIS ACCORDING TO MAPPING FILE IN THE GITHUB OF MATTERPORT3D
/*
const std::map<std::string, std::vector<int>> Segmentation_Provider::label_to_color = { 
    { "void", {255, 0, 255} }, 
    { "wall", {255, 0, 255} },
    { "floor", {255, 0, 255} },
    { "chair", {255, 0, 255} },
    { "door", {255, 0, 255} },
    { "table", {255, 0, 255} }, 
    { "picture", {255, 0, 255} },
    { "cabinet", {255, 0, 255} },
    { "cushion", {255, 0, 255} },
    { "window", {255, 0, 255} },
};
*/