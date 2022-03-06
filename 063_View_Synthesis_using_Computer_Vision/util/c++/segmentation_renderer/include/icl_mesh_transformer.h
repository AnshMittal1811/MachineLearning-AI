#pragma once

#include "mesh.h"
#include <glm/glm.hpp>

class ICL_Mesh_Transformer {

    public:
        ICL_Mesh_Transformer(Mesh& mesh);
        ~ICL_Mesh_Transformer();

        //void splitMeshAtObject(int objectId);
        void moveVerticesOfObject(std::string name, glm::mat4& transformation);
    
    private:
        Mesh m_mesh;
};