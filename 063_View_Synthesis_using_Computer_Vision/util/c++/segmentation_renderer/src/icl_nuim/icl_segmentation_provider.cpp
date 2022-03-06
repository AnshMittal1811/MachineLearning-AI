#include "icl_segmentation_provider.h"

ICL_Segmentation_Provider::ICL_Segmentation_Provider(string const &object_to_color_path) {
    // read object_to_color file
    std::ifstream f(object_to_color_path);
    
    if(! f.is_open()){
        throw std::runtime_error("Could not read object_to_color file from: " + object_to_color_path);
    }

    std::string name;
    float r, g, b;

    while(f >> name >> r >> g >> b){
        object_name_to_color[name] = glm::vec3(r, g, b);
    }
}

ICL_Segmentation_Provider::~ICL_Segmentation_Provider() = default;

void ICL_Segmentation_Provider::change_colors(Mesh &mesh){
    for(auto i=0; i<mesh.vertices.size(); i++){

        // look up the new color for this vertex
        glm::vec3 rgb = object_name_to_color[mesh.vertices[i].Name];

        // assign new color
        mesh.vertices[i].Color = rgb;
    }

    // update information on GPU for OpenGL to render with new values
    mesh.updateData();
}

json ICL_Segmentation_Provider::getMovementAsJson(std::string name, glm::mat4& transformation){
    json j;

    // FILL COLOR
    glm::vec3 color = object_name_to_color[name];
    std::vector<float> color_vec;
    color_vec.emplace_back(color[0]);
    color_vec.emplace_back(color[1]);
    color_vec.emplace_back(color[2]);
    j["color"] = color_vec;

    // FILL NAME
    j["name"] = name;

    // FILL TRANSFORMATION IN ROW MAYOR ORDER AND ONLY KEEP THE 3x4 RT MATRIX WITHOUT [0 0 0 1] LINE
    std::vector<float> trans_vec;
    trans_vec.emplace_back(transformation[0][0]);
    trans_vec.emplace_back(transformation[1][0]);
    trans_vec.emplace_back(transformation[2][0]);
    trans_vec.emplace_back(transformation[3][0]);

    trans_vec.emplace_back(transformation[0][1]);
    trans_vec.emplace_back(transformation[1][1]);
    trans_vec.emplace_back(transformation[2][1]);
    trans_vec.emplace_back(transformation[3][1]);

    trans_vec.emplace_back(transformation[0][2]);
    trans_vec.emplace_back(transformation[1][2]);
    trans_vec.emplace_back(transformation[2][2]);
    trans_vec.emplace_back(transformation[3][2]);

    j["transformation"] = trans_vec;

    return j;
}