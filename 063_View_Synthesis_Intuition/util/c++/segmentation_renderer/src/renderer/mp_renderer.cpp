#include "mp_renderer.h"
#include "util.h"

MP_Renderer::MP_Renderer(string const &pathToMesh, MP_Parser const &mp_parser, int region_index): 
                    mp_parser(mp_parser),
                    region_index(region_index),
                    Renderer(pathToMesh,
                             mp_parser.regions[region_index]->panoramas[0]->images[0]->width,
                             mp_parser.regions[region_index]->panoramas[0]->images[0]->height) {

}

MP_Renderer::~MP_Renderer() = default;

/*
    - The region_X.ply are still in world-coordinates, e.g. region0 is left and region6 is centered.
    - Thus I can use the camera extrinsics/intrinsics also for the regions only
    - This means, that I can use regions + vseg file (Alternative: use whole house mesh and parse fseg file instead of vseg)
    - For each image (matterport_color_images.zip) we have a corresponding extrinsic/intrinsic file with same name
        --> Use this for calculating the view and projection matrices
        --> But these parameters are distorted, e.g. the intrinsic files contain arbitrary 3x3 matrix
        --> This is solved in undistorted_camera_parameters.zip
        --> The same values as in undistorted_camera_parameters.zip are also present in the .house file
        --> Just use the extrinsic/intrinsic parameters from the .house file!
        --> Note that the extrinsic parameters differ in the .house file and in the undistorted file. What is correct?
    - Find out which image corresponds to which region. It only makes sense to use the images for the corresponding region
        --> Otherwise we would look at nothing because in that case the region is not present
        --> Can I do it like this? Parse .house file and go like this: Image Name --> Panorama Index --> Region Index ? --> Yes!
*/
void MP_Renderer::renderImages(const std::string save_path){

    for(int i=0; i<mp_parser.regions[region_index]->panoramas.size(); i++){
        for(MPImage* image : mp_parser.regions[region_index]->panoramas[i]->images){

            glm::mat4 extr = glm::transpose(glm::make_mat4(image->extrinsics));
            glm::mat3 intr = glm::make_mat3(image->intrinsics);
            glm::mat4 projection = camera_utils::perspective(intr, image->width, image->height, kNearPlane, kFarPlane);

            // render image
            render(glm::mat4(1.0f), extr, projection);

            // read image into openCV matrix
            cv::Mat colorImage;
            readRGB(colorImage);

            // save matrix as file
            if ((save_path != "") && (!colorImage.empty())) {
                std::stringstream filename;
                filename << save_path << "/segmentation_" << image->color_filename;
                cv::imwrite(filename.str(), colorImage);

                std::cout << "Wrote segmentation of: " << image->color_filename << std::endl;
            }

            // show image in window
            glfwSwapBuffers(m_window);

        }
    
    }
}