import seaborn as sns
import numpy as np
import open3d as o3d

'''
Author: Rahul Sajnani
Date: 16th September 2021
'''


def create_color_samples(N):
    '''
    Creates N distinct colors
    N x 3 output
    '''

    palette = sns.color_palette(None, N)
    palette = np.array(palette)

    return palette


def visualize_pointclouds(pcd_list, colors_list = None):
    '''
    Visualize the list of point clouds in Open3D
    size = 3 x N
    '''
    pallete = create_color_samples(len(pcd_list))
    pcd_object_list = []

    for pcd_num in range(len(pcd_list)):

        points = pcd_list[pcd_num]
        if colors_list is not None:
            colors = colors_list[pcd_num]
        else:
            colors = np.ones(points.shape) * pallete[pcd_num, :][:, np.newaxis]
        
        pcd_cloud = o3d.geometry.PointCloud()
        pcd_cloud.points = o3d.utility.Vector3dVector(points.T)
        pcd_cloud.colors = o3d.utility.Vector3dVector(colors.T)

        pcd_object_list.append(pcd_cloud)


    o3d.visualization.draw_geometries(pcd_object_list)
