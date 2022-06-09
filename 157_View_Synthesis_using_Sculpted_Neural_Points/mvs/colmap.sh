#!/bin/bash
colmap_install_dir=$1
INPUT_DIR=$2

$colmap_install_dir feature_extractor  --database_path $INPUT_DIR/db.db --image_path $INPUT_DIR/images --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_PINHOLE
$colmap_install_dir exhaustive_matcher --database_path $INPUT_DIR/db.db  --SiftMatching.guided_matching 1
mkdir -p $INPUT_DIR/sparse
$colmap_install_dir mapper --database_path $INPUT_DIR/db.db --image_path $INPUT_DIR/images --output_path $INPUT_DIR/sparse --Mapper.init_min_tri_angle 4 --Mapper.num_threads 16 --Mapper.multiple_models 0 --Mapper.extract_colors 0

mkdir -p $INPUT_DIR/dense
$colmap_install_dir image_undistorter --image_path $INPUT_DIR/images --input_path $INPUT_DIR/sparse/0 --output_path $INPUT_DIR/dense --output_type COLMAP
$colmap_install_dir patch_match_stereo --workspace_path $INPUT_DIR/dense --workspace_format COLMAP --PatchMatchStereo.max_image_size 1600
$colmap_install_dir stereo_fusion --workspace_path $INPUT_DIR/dense --workspace_format COLMAP --input_type geometric --output_path $INPUT_DIR/dense/fused.ply --StereoFusion.min_num_pixels 2 --StereoFusion.max_image_size 1600

