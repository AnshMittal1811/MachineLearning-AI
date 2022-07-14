#include "segmentation_provider.h"
#include <set>

Segmentation_Provider::Segmentation_Provider(string const &semseg_path,
                                             string const &vseg_path,
                                             string const &fseg_path,
                                             Mesh &mesh) {
    // read semseg file
    std::ifstream semseg_file(semseg_path);
    semseg_file >> semseg;

    // read vseg file
    std::ifstream vseg_file(vseg_path);
    vseg_file >> vseg;

    // read fseg file
    std::ifstream fseg_file(fseg_path);
    fseg_file >> fseg;

    // Number of Vertices as defined in the vseg file
    n_vertices = vseg["segIndices"].size();

    // Number of objects as defined in the semseg file
    n_objects = semseg["segGroups"].size();
    
    // Vertex to Segment map
    for (auto& segment : vseg["segIndices"]) {
        // the json file contains the segments listed in order of the vertices, so first entry is segment of first vertex
        vertex_to_segment.push_back(segment);
    }

    // Face to Segment map
    for (auto& segment : fseg["segIndices"]) {
        // the json file contains the segments listed in order of the faces, so first entry is segment of first face
        face_to_segment.push_back(segment);
    }

    // Vertex to Face map
    for(auto i=0; i<mesh.indices.size(); i++){

        int face = i / 3; // true division to get the face index, i.e. the first 3 indices all correspond to the first face, etc.
        
        /* Since vertices can be part of multiple faces, we need to choose any of these faces as THE face for each vertex.
           With this implementaion, the face of the vertex will always be the face that appears last in the meshes indices list for this vertex.

           TODO: Check if the generation of the .vseg file is consistent with that!! */
        int vertex = mesh.indices[i];
        
        if(vertex_to_face.find(vertex) == vertex_to_face.end())
            vertex_to_face[vertex] = face;
    }

    // DEBUG: Check consistency of vertex_to_face + face_to_segment VS. vertex_to_segment directly
    for(auto i=0; i<mesh.vertices.size(); i++){
        int s1 = vertex_to_segment[i];
        int s2 = face_to_segment[vertex_to_face[i]];

        if(s1 != s2){
            //std::cerr << "Inconsistent segments for vertex " << i << ": " << s1 << " vs, " << s2 << std::endl;
        }
    }

    // Segment to Object map
    for (auto& group : semseg["segGroups"]) {
        int id = group["id"];
        // read all segments for this group
        for (auto& segment : group["segments"]) {
            segment_to_object_id[segment] = id;
        }
    }

    // Object to Color Map
    for (auto i=0; i<n_objects; i++) {
        float rgb[3];
        random_rgb(rgb);
        object_id_to_color[i] = glm::vec3(rgb[0], rgb[1], rgb[2]);
    }

    // Segment to Class map
    std::set<std::string> labels;

    for (auto& group : semseg["segGroups"]) {
        std::string label = group["label"];
        labels.insert(label);
        // read all segments for this group
        for (auto& segment : group["segments"]) {
            segment_to_class[segment] = label;
        }
    }

    // Class to Color Map
    for (auto& c : labels) {
        float rgb[3];
        random_rgb(rgb);
        class_to_color[c] = glm::vec3(rgb[0], rgb[1], rgb[2]);
    }
}

Segmentation_Provider::~Segmentation_Provider() = default;

int Segmentation_Provider::getObjectId(int vertexIndex){
    int segment = vertex_to_segment[vertexIndex];
    int object_id = segment_to_object_id[segment];

    int s2 = face_to_segment[vertex_to_face[vertexIndex]];
    //object_id = segment_to_object_id[s2];

    return object_id;
}

void Segmentation_Provider::change_colors(Mesh &mesh){

    std::cout << "fsegs number of segments: " << fseg["segIndices"].size() << std::endl;
    std::cout << "mesh number of indices (3x number segments if only triangles in .ply file): " << mesh.indices.size() << std::endl;

    for(auto i=0; i<mesh.vertices.size(); i++){

        // look up the new color for this vertex
        int object_id = getObjectId(i);
        glm::vec3 rgb = object_id_to_color[object_id];

        // assign new color
        mesh.vertices[i].Color = rgb;
    }

    // update information on GPU for OpenGL to render with new values
    mesh.updateData();
}