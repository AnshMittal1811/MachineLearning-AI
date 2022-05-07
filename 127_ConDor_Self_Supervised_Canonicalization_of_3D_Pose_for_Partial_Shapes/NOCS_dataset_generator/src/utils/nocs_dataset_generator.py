import numpy as np
import sys
import nocs, glob, vis, sampling, os, h5py, draco_utils, cv2
from sklearn.neighbors import NearestNeighbors
import concurrent.futures


def generate_dataset_DRACO(input_folder, num_points):
    """
    Generates DRACO dataset with camera posese and depth maps
    """
    nocs_files = glob.glob(input_folder + "**_NOXRayTL_00.png")
    pose_files = glob.glob(input_folder + "**_CameraPose.json")
    depth_files = glob.glob(input_folder + "**_Depth_00.exr")

    nocs_files.sort()
    pose_files.sort()
    depth_files.sort()

    pcd_list = []
    pcd_depth_list = []
    pcd_list_fps = []
    pose_list = []
    idx_list = []

    for i in range(len(nocs_files)):
        
        file_name = nocs_files[i]
        depth_name = depth_files[i]
        pose_name = pose_files[i]

        nocs_image = nocs.read_NOCS_image(file_name)
        nocs_mask = nocs.generate_NOCS_mask(nocs_image)
        pcd = nocs.extract_NOCS_pointcloud(nocs_image, min_points=6000)
        depth_map = cv2.imread(depth_name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
        pcd_depth = draco_utils.extract_depth_pointcloud(depth_map, nocs_mask)
        
        pose_json = draco_utils.read_json_file(pose_name)
        pose_mat = draco_utils.pose_to_mat(pose_json)

        pcd_sampled, idx_samples = sampling.fps(pcd, num_points)
        
        pose_list.append(pose_mat)
        pcd_list.append(pcd)
        pcd_depth_sampled = pcd_depth[idx_samples, :]#, _ = sampling.fps(pcd_depth, num_points)
        # print("depth",pcd_depth_sampled.shape, "nocs", pcd_sampled.shape)
        # check_mat, _, _ = draco_utils.solve_orthogonal_procrustes(pcd_sampled.T - 0.5, pcd_depth_sampled.T)
        # print(check_mat , " Computed\n", pose_mat, "GT")
        
        pcd_depth_list.append(pcd_depth_sampled)
        pcd_list_fps.append(pcd_sampled)

    pcd_final_full = np.vstack(pcd_list)
    pcd_depth_final = np.stack(pcd_depth_list, axis = 0)
    pose_final = np.stack(pose_list, axis=0)

    # print(pcd_depth_final.shape)

    points_sparse = range(0, pcd_final_full.shape[0], 50)
    pcd_final_full = pcd_final_full[points_sparse, :]
    pcd_final_full, idx = sampling.fps(pcd_final_full, num_points * 4)
    # vis.visualize_pointclouds([pcd_final_full.T], [pcd_final_full.T])

    nbrs = NearestNeighbors(
        n_neighbors=1, algorithm='ball_tree').fit(pcd_final_full)

    for pcd in pcd_list_fps:
        distances, idx = nbrs.kneighbors(pcd)
        idx_list.append(idx.ravel())

    stacked_partial_matrix = np.stack(pcd_list_fps, axis=0)
    # stacked_depth = np.stack(depth)
    stacked_full_matrix = np.stack(
        [pcd_final_full] * stacked_partial_matrix.shape[0], axis=0)
    stacked_full_mean = np.mean(stacked_full_matrix, axis=1, keepdims=True)

    # Mean center the dataset
    stacked_full_matrix = stacked_full_matrix - stacked_full_mean
    stacked_partial_matrix = stacked_partial_matrix - stacked_full_mean
    stacked_idx_matrix = np.stack(idx_list, axis=0)

    return stacked_partial_matrix, stacked_full_matrix, stacked_idx_matrix, pcd_depth_final, pose_final


def generate_dataset_folder(input_folder, num_points):
    '''
    Generates a pointcloud dataset to be used for training

    '''

    nocs_files = glob.glob(input_folder + "**_NOXRayTL_00.png")
    nocs_files.sort()

    pcd_list = []
    pcd_list_fps = []
    idx_list = []


    for file_name in nocs_files:

        # print(file_name)
        nocs_image = nocs.read_NOCS_image(file_name)
        pcd = nocs.extract_NOCS_pointcloud(nocs_image, min_points = 6000)
        pcd_list.append(pcd)
        pcd_sampled, idx_samples = sampling.fps(pcd, num_points)
        pcd_list_fps.append(pcd_sampled)

    pcd_final_full = np.vstack(pcd_list)

    points_sparse = range(0, pcd_final_full.shape[0],50)
    pcd_final_full = pcd_final_full[points_sparse, :]
    pcd_final_full, idx = sampling.fps(pcd_final_full, num_points * 4)
    # vis.visualize_pointclouds([pcd_final_full.T], [pcd_final_full.T])

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(pcd_final_full)

    for pcd in pcd_list_fps:
        distances, idx = nbrs.kneighbors(pcd)
        idx_list.append(idx.ravel())

    stacked_partial_matrix = np.stack(pcd_list_fps, axis = 0)
    stacked_full_matrix = np.stack([pcd_final_full]* stacked_partial_matrix.shape[0], axis = 0)
    stacked_full_mean = np.mean(stacked_full_matrix, axis = 1, keepdims=True)

    # Mean center the dataset
    stacked_full_matrix = stacked_full_matrix - stacked_full_mean
    stacked_partial_matrix = stacked_partial_matrix - stacked_full_mean
    stacked_idx_matrix = np.stack(idx_list, axis = 0)

    return stacked_partial_matrix, stacked_full_matrix, stacked_idx_matrix

def generate_dataset_category(input_dir, output_h5_file, num_points, num_processes = 4, save_freq = 2, max_folders = 900, draco = 0):
    '''
    Generate point cloud dataset for a category
    '''

        
    dataset_file = h5py.File(output_h5_file, "w")
    input_directory = os.path.join(input_dir)

    folder_list = [os.path.join(input_directory, o) for o in os.listdir(input_directory)
                    if os.path.isdir(os.path.join(input_directory,o))]

    dataset_partial = []
    dataset_full = []
    dataset_idx = []
    dataset_depth = []
    dataset_pose = []

    counter = 0
    first_save = 1
    folder_list = [os.path.join(folder, "") for folder in folder_list]
    folder_list = folder_list[:max_folders]   
    with concurrent.futures.ProcessPoolExecutor(max_workers = num_processes) as executor:
        
        num_points_map = [num_points] * len(folder_list)
        if draco:
            results = executor.map(generate_dataset_DRACO, folder_list, num_points_map)
        else:
            results = executor.map(
                generate_dataset_folder, folder_list, num_points_map)

        for result in results:
            if draco:
                stacked_partial_matrix, stacked_full_matrix, stacked_idx, depth, pose = result
                dataset_depth.append(depth)
                dataset_pose.append(pose)
            else:
                stacked_partial_matrix, stacked_full_matrix, stacked_idx = result    
            
            dataset_partial.append(stacked_partial_matrix)
            dataset_full.append(stacked_full_matrix) 
            dataset_idx.append(stacked_idx)

            counter += 1
            print(counter)

    
    dataset_full_save = np.vstack(dataset_full)
    dataset_partial_save  = np.vstack(dataset_partial)
    dataset_idx_save = np.vstack(dataset_idx)

    dataset_dict = {"full": dataset_full_save, "partial": dataset_partial_save, "idx": dataset_idx_save}
    if draco:
        dataset_depth_save = np.vstack(dataset_depth)
        dataset_pose_save = np.vstack(dataset_pose)
        dataset_dict["depth"] = dataset_depth_save
        dataset_dict["pose"] = dataset_pose_save

    for key in dataset_dict.keys():
        print("key", key, " with shape ", dataset_dict[key].shape)
        dataset_file.create_dataset(key, data = dataset_dict[key])

    dataset_file.close()

if __name__ == "__main__":

    # input_dir = "/home/rahul/Internship/Brown2021/code/EquiNet/NOCS-dataset-generator/src/check"
    input_dir = "/home/rahul/Education/success/train_actual_small"
    # input_dir = "/home/rahul/Education/success/dataset_check"

    #generate_dataset_category(input_dir, "./cars.hdf5", 1024, num_processes = 2 )
    generate_dataset_category(input_dir, "./cars_draco.h5", 1024, num_processes=2, draco = 1)

    # print(stacked_partial_matrix.shape, stacked_full_matrix.shape)
