#pragma once

#include "mesh.h"

#include <map>
#include <glm/glm.hpp>


#include "json.hpp"
// for convenience
using json = nlohmann::json;

class ICL_Segmentation_Provider {

    public:
        ICL_Segmentation_Provider(string const &object_to_color_path);
        ~ICL_Segmentation_Provider();
        void change_colors(Mesh &mesh);
        json getMovementAsJson(std::string name, glm::mat4& transformation);
    
    private:
        std::map<std::string, glm::vec3> object_name_to_color;

};