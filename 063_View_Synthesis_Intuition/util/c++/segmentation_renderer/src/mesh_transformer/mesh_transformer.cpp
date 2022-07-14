#include "mesh_transformer.h"

Mesh_Transformer::Mesh_Transformer(Mesh& mesh, Segmentation_Provider& sp):
                                    m_mesh(mesh), m_sp(sp) { }

Mesh_Transformer::~Mesh_Transformer() = default;

void Mesh_Transformer::splitMeshAtObject(int objectId) {
    std::vector<unsigned int> newIndices;

    // check each triangle if it is on the same object or not
    // If on different objects AND one of the objects is the object to split: remove this triangle
    for(int i=0; i< m_mesh.indices.size(); i+=3) {
        int ids[3];
        ids[0] = m_sp.getObjectId(m_mesh.indices[i]);
        ids[1] = m_sp.getObjectId(m_mesh.indices[i+1]);
        ids[2] = m_sp.getObjectId(m_mesh.indices[i+2]);

        bool sameObjectIds = ids[0] == ids[1] && ids[1] == ids[2];
        bool noneOfTargetObjectId = ids[0] != objectId && ids[1] != objectId && ids[2] != objectId;

        if(sameObjectIds || noneOfTargetObjectId){
            // can keep all of these indices... the other ones are discarded because they match the above criterion
            newIndices.push_back(m_mesh.indices[i]);
            newIndices.push_back(m_mesh.indices[i+1]);
            newIndices.push_back(m_mesh.indices[i+2]);
        }
    }

    m_mesh.indices = newIndices; // TODO because we completely replace the original indices, we can only ever do one split per mesh load!!
    m_mesh.updateData();
}

void Mesh_Transformer::moveVerticesOfObject(int objectId, glm::mat4& transformation){
    std::vector<Vertex> newVertices;

    for(int i=0; i<m_mesh.vertices.size(); i++) {
        int id;
        id = m_sp.getObjectId(i);

        if(id == objectId){
            Vertex v = m_mesh.vertices[i];

            v.Position = transformation * glm::vec4(v.Position, 1.0f);

            glm::mat3 it_trans = glm::transpose(glm::inverse(transformation));
            v.Normal = it_trans * v.Normal;
            v.Tangent = it_trans * v.Tangent; // TODO NOT SURE ABOUT THIS
            v.Bitangent = it_trans * v.Bitangent; // TODO NOT SURE ABOUT THIS

            newVertices.push_back(v);

        } else {
            newVertices.push_back(m_mesh.vertices[i]);
        }
    }

    m_mesh.vertices = newVertices;
    m_mesh.updateData();
}