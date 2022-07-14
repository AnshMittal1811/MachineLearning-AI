#include <iostream>

//#include "icl_renderer.h"
//#include "icl_parser.h"
//#include "icl_mesh_transformer.h"
//#include "icl_segmentation_provider.h"

#include "mp_renderer.h"
#include "mp_parser.h"
#include "segmentation_provider.h"
#include "mesh_transformer.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include <glm/glm.hpp>

/*
        - TODO: Write a main.cpp pipeline which loops over all images (selected images --> loop over folder of images) for a specific region
            --> Color according to segmentation + transform object from input
            --> Render from specific camera pose/intrinsic for this view
            --> Save as image in output folder
        - TODO (optional): Before the main.cpp pipeline starts, we show the region in an interactive renderer.
            --> Allow the user to somehow interactively choose which object to move and how to move it
            --> From this selection, extract a transformation matrix and use that as an input for the pipeline
            --> (optional 2): Let the user create a trajectory (multiple transformation matrices) and use each of them
*/

int main(int argc, char** argv){

    /*
    std::string seq = "seq0019";

    // ICL_Parser ip("/home/lukas/Desktop/datasets/ICL-NUIM/prerendered_data/living_room_traj2_loop", 0);
    ICL_Parser ip("/home/lukas/Desktop/datasets/ICL-NUIM/custom/" + seq + "/original", 0);

    ICL_Renderer icl_renderer("/home/lukas/Desktop/datasets/ICL-NUIM/model_for_rendering/living_room_obj_mtl/living-room.obj");

    ICL_Segmentation_Provider icl_sp("../src/icl_nuim/object_to_color.txt");

    glm::mat4 t(1.0f);
    t = glm::translate(t, glm::vec3(-0.15f, 0.0f, -0.26f)); // for seq0001
    // t = glm::rotate(t, glm::radians(20.0f), glm::vec3(0.0f, 1.0f, 0.0f)); // for seq0002
    t = glm::rotate(t, glm::radians(25.0f), glm::vec3(1.0f, 1.0f, 0.0f));
    std::string move_object = "cushion3_cushion3";

    std::cout << glm::to_string(t) << std::endl;

    for(auto& mesh : icl_renderer.m_model->meshes){
        icl_sp.change_colors(mesh);
    }

    // icl_renderer.renderInteractive(ip);

    // RENDER ORIGINAL
    icl_renderer.renderTrajectory(ip, "/home/lukas/Desktop/datasets/ICL-NUIM/custom/" + seq + "/original");

    // RENDER MOVED

    for(auto& mesh : icl_renderer.m_model->meshes){
        ICL_Mesh_Transformer icl_mt(mesh);
        icl_mt.moveVerticesOfObject(move_object, t);
    }

    json moved_json;
    moved_json.push_back(icl_sp.getMovementAsJson(move_object, t));
    std::ofstream moved_file("/home/lukas/Desktop/datasets/ICL-NUIM/custom/" + seq + "/moved.txt");
    moved_file << std::setw(4) << moved_json << std::endl;

    //Model icl_model("/home/lukas/Desktop/datasets/ICL-NUIM/model_for_rendering/living_room_obj_mtl/living-room.obj");
    icl_renderer.renderTrajectory(ip, "/home/lukas/Desktop/datasets/ICL-NUIM/custom/" + seq + "/moved");
    // icl_renderer.renderTrajectory(ip, "/home/lukas/Desktop/datasets/ICL-NUIM/prerendered_data/living_room_traj2_loop");
    */
    
    
    if(argc != 3){
        std::cout << "Usage: " << argv[0] << " path/to/Matterport3D/data/v1/scans <scanID>" << std::endl;
        return EXIT_FAILURE;
    }

    string path(argv[1]);
    string scanID(argv[2]);
    string outdir = path + "/" + scanID + "/image_segmentations"; // TODO mkdir this

    string pathToHouseFile = path + "/" + scanID + "/house_segmentations/" + scanID + "/house_segmentations/" + scanID + ".house";
    MP_Parser mp(pathToHouseFile.c_str());
    std::cout << "parsed .house file" << std::endl;

    try{
        string regionPath = path + "/" + scanID + "/region_segmentations/" + scanID + "/region_segmentations";

        regionPath += "/region0."; // TODO instead loop over all regions!

        MP_Renderer renderer(regionPath + "ply", mp, 0);

        std::cout << "Renderer initialized" << std::endl;

        for(auto& mesh : renderer.m_model->meshes){
            Segmentation_Provider sp(regionPath + "semseg.json", regionPath + "vsegs.json", regionPath + "fsegs.json", mesh);
            sp.change_colors(mesh);
        }

        std::cout << "Updated Mesh Colors according to object instance segmentation" << std::endl;

        // render original (before move)
        renderer.renderImages(outdir + "/original");

        for(auto& mesh : renderer.m_model->meshes){
            Segmentation_Provider sp(regionPath + "semseg.json", regionPath + "vsegs.json", regionPath + "fsegs.json", mesh);
            Mesh_Transformer transform(mesh, sp);
            transform.splitMeshAtObject(1);
            glm::mat4 t = { 1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, -0.5, 0.5, 1 };
            transform.moveVerticesOfObject(1, t);
        }

        std::cout << "Splitted and transformed mesh: pillow with id 1 translated by (0, -0.5, 0.5)" << std::endl;

        // render moved
        renderer.renderImages(outdir + "/moved");

        std::cout << "Render images completed" << std::endl;

        // renderer.renderInteractive();

    } catch(const exception& e){
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    

    return EXIT_SUCCESS;
}