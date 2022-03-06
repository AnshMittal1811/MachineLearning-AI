#include "icl_mesh_transformer.h"

ICL_Mesh_Transformer::ICL_Mesh_Transformer(Mesh& mesh):
                                    m_mesh(mesh) { }

ICL_Mesh_Transformer::~ICL_Mesh_Transformer() = default;

void ICL_Mesh_Transformer::moveVerticesOfObject(std::string name, glm::mat4& transformation){
    std::vector<Vertex> newVertices;

    for(int i=0; i<m_mesh.vertices.size(); i++) {
        Vertex v = m_mesh.vertices[i];
        string vertexName = v.Name;

        if(vertexName == name){
            v.Position = transformation * glm::vec4(v.Position, 1.0f);

            glm::mat3 it_trans = glm::transpose(glm::inverse(transformation));
            v.Normal = it_trans * v.Normal;
            v.Tangent = it_trans * v.Tangent; // TODO NOT SURE ABOUT THIS
            v.Bitangent = it_trans * v.Bitangent; // TODO NOT SURE ABOUT THIS

            newVertices.push_back(v);

        } else {
            newVertices.push_back(v);
        }
    }

    m_mesh.vertices = newVertices;
    m_mesh.updateData();
}