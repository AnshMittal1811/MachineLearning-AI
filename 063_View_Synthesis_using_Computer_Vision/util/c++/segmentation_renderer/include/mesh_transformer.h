#pragma once

#include "mesh.h"
#include "segmentation_provider.h"
#include <glm/glm.hpp>

class Mesh_Transformer {

    public:
        Mesh_Transformer(Mesh& mesh, Segmentation_Provider& sp);
        ~Mesh_Transformer();

        void splitMeshAtObject(int objectId);
        void moveVerticesOfObject(int objectId, glm::mat4& transformation);
    
    private:
        Mesh m_mesh;
        Segmentation_Provider m_sp;
};