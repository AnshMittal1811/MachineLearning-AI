# Segmentation Renderer
The segmentation renderer allows to render instance segmentation masks out of the original Matterport3D camera images. You can also choose to move certain objects of the mesh arouny by an arbitrary transformation matrix. 

## Prerequisites
- at least c++11 capable compiler (gcc, ...)
   - Check if installed: gcc --version
- cmake
   - sudo apt-get install cmake
- At least the following 3 directories unzipped for each Matterport3D scan that shall be used:
   - matterport_color_images
   - region_segmentations
   - house_segmentations
- With the current version, these two steps are still necessary (will be done automatically later...)
   - Create a new directory "image_segmentations" in the root of each Matterport3D scan to use.
   - Create two subdirectories "original", "moved" in the "image_segmentations" folder.

## Setup (Dependencies for this program)

    sudo apt-get install assimp-utils libassimp-dev
    sudo apt-get install libopencv-dev
    sudo apt-get install libglm-dev
    sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
    sudo apt-get install libglfw3-dev libglfw3
    sudo apt-get install libglew-dev

## Build the Program
    mkdir build
    cd build
    cmake ..
    make

## Run the Program
    ./segmentation_renderer /path/to/Matterport3D/data/v1/scans <scanID>

    ./segmentation_renderer ~/datasets/Matterport3D/data/v1/scans 17DRP5sb8fyD
    
Note that the program has hardcoded choices: 
   - The region0 will be rendered from all available views.
   - The original and moved views will be rendered as segmentation masks
   - Moving is currently hardcoded as: Move object_id 1 in region0 by a translation vector of (0, -0.5, 0.5).

You might edit these choices by modifying main.cpp accordingly.
Later, the program might become more flexible.
