#pragma once

#include <iostream>
#include <string>

#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <cstring>
#include <dirent.h>

#include <glm/glm.hpp>

class ICL_Parser {
public:
    ICL_Parser(std::string _filebasename, int _scene_nr): filebasename(_filebasename), scene_nr(_scene_nr) {
        // Parse directory
        int len;
        struct dirent *pDirent;
        DIR *pDir = NULL;

        txtfilecount = 0;
        pngfilecount = 0;
        depthfilecount = 0;

        pDir = opendir(filebasename.c_str());
        if (pDir != NULL)
        {
            while ((pDirent = readdir(pDir)) != NULL)
            {
                len = strlen (pDirent->d_name);
                if (len >= 4)
                {
                    if (strcmp (".txt", &(pDirent->d_name[len - 4])) == 0)
                    {
                        txtfilecount++;
                    }
                    else if (strcmp (".png", &(pDirent->d_name[len - 4])) == 0)
                    {
                        pngfilecount++;
                    }
                    else if (strcmp (".depth", &(pDirent->d_name[len - 6])) == 0)
                    {
                        depthfilecount++;
                    }
                }
            }
            closedir (pDir);
        }

        if ( txtfilecount != pngfilecount || txtfilecount != depthfilecount || pngfilecount != depthfilecount) {
            std::cerr << "Warning: The number of depth files (" << depthfilecount << "), png files (" << pngfilecount << ") and txt files (" << txtfilecount << ") are not same." << std::endl;
        }

    }

    // Obtain the pose of camera Tpov_cam, with respect to povray world
    glm::mat4 getExtrinsics(int ref_img_no){
        char text_file_name[360];
        sprintf(text_file_name, "%s/scene_%02d_%04d.txt",
                filebasename.c_str(), scene_nr, ref_img_no);
        std::ifstream cam_pars_file(text_file_name);
        char readlinedata[300];

        glm::vec4 direction;
        glm::vec4 upvector;
        glm::vec3 posvector;


        while(1)
        {
            cam_pars_file.getline(readlinedata,300);

            if ( cam_pars_file.eof()){
                break;
            }

            std::istringstream iss;

            if ( strstr(readlinedata,"cam_dir")!= NULL)
            {
                std::string cam_dir_str(readlinedata);

                cam_dir_str = cam_dir_str.substr(cam_dir_str.find("= [")+3);
                cam_dir_str = cam_dir_str.substr(0,cam_dir_str.find("]"));

                iss.str(cam_dir_str);
                iss >> direction.x;
                iss.ignore(1,',');
                iss >> direction.y;
                iss.ignore(1,',') ;
                iss >> direction.z;
                iss.ignore(1,',');
                //cout << direction.x<< ", "<< direction.y << ", "<< direction.z << endl;
                direction.w = 0.0f;

            }

            if ( strstr(readlinedata,"cam_up")!= NULL)
            {

                std::string cam_up_str(readlinedata);

                cam_up_str = cam_up_str.substr(cam_up_str.find("= [")+3);
                cam_up_str = cam_up_str.substr(0,cam_up_str.find("]"));


                iss.str(cam_up_str);
                iss >> upvector.x;
                iss.ignore(1,',');
                iss >> upvector.y;
                iss.ignore(1,',');
                iss >> upvector.z;
                iss.ignore(1,',');


                upvector.w = 0.0f;

            }

            if ( strstr(readlinedata,"cam_pos")!= NULL)
            {
                std::string cam_pos_str(readlinedata);

                cam_pos_str = cam_pos_str.substr(cam_pos_str.find("= [")+3);
                cam_pos_str = cam_pos_str.substr(0,cam_pos_str.find("]"));

                iss.str(cam_pos_str);
                iss >> posvector[0];
                iss.ignore(1,',');
                iss >> posvector[1];
                iss.ignore(1,',');
                iss >> posvector[2];
                iss.ignore(1,',');

            }

        }

        /// z = dir / norm(dir)
        glm::vec3 z;
        z[0] = direction.x;
        z[1] = direction.y;
        z[2] = direction.z;
        z = glm::normalize(z);

        /// x = cross(cam_up, z)
        glm::vec3 x;
        x[0] =  upvector.y * z[2] - upvector.z * z[1];
        x[1] =  upvector.z * z[0] - upvector.x * z[2];
        x[2] =  upvector.x * z[1] - upvector.y * z[0];
        x = glm::normalize(x);

        /// y = cross(z,x)
        glm::vec3 y;
        y[0] =  z[1] * x[2] - z[2] * x[1];
        y[1] =  z[2] * x[0] - z[0] * x[2];
        y[2] =  z[0] * x[1] - z[1] * x[0];

        /// contstruct [R|T]
        glm::mat4 RT(0.0);
        RT[0][0] = x[0];
        RT[1][0] = x[1];
        RT[2][0] = x[2];
        RT[3][0] = 0.0;

        RT[0][1] = y[0];
        RT[1][1] = y[1];
        RT[2][1] = y[2];
        RT[3][1] = 0.0;

        RT[0][2] = z[0];
        RT[1][2] = z[1];
        RT[2][2] = z[2];
        RT[3][2] = 0.0;

        RT[0][3] = posvector.x;
        RT[1][3] = posvector.y;
        RT[2][3] = posvector.z;
        RT[3][3] = 1.0;


        return glm::transpose(RT); // fkin column-mayor shit
    }

    glm::mat3 getIntrinsics(){
        glm::mat3 K(0.0);
        K[0][0] = fx;
        K[1][1] = fy;
        K[0][2] = cx;
        K[1][2] = cy;
        K[2][2] = 1;

        //return glm::transpose(K); // fkin column-mayor shit

        return K;
    }

    // Get the number of relevant files
    int getNumberofPoseFiles(){
        return txtfilecount; 
    }
    int getNumberofImageFiles(){
        return pngfilecount; 
    }
    int getNumberofDepthFiles(){
        return depthfilecount;
    }
    int getWidth(){
        return ICL_WIDTH;
    }
    int getHeight(){
        return ICL_HEIGHT;
    }
    int getSceneNr(){
        return scene_nr;
    }

private:
    // Directory variables
    std::string filebasename;
    int scene_nr;
    int txtfilecount;
    int pngfilecount;
    int depthfilecount;

    unsigned int ICL_WIDTH = 640;
    unsigned int ICL_HEIGHT = 480;

    float fx = 481.2;
    float fy = -480.0; // OR USE -480.0?
    float cx = 319.5;
    float cy = 239.5;
};