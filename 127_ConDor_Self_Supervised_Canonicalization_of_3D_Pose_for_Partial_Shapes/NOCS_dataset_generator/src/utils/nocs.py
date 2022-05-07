import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
import vis

def read_NOCS_image(file_path):
    '''
    Read the NOCS image
    '''

    image = plt.imread(file_path)

    if image.max() > 1:
        image = image / image.max()

    return image[:, :, :3]

def generate_NOCS_mask(image):
    '''
    Generate mask from NOCS image
    '''

    mask = np.ones(image.shape[:2])
    for channel_num in range(image.shape[2]):
        mask *= (image[:, :, channel_num] != 1)
    return (mask == 1)


def extract_NOCS_pointcloud(image, min_points = 10000):
    '''
    Obtain NOCS point cloud from NOCS image
    '''

    mask = generate_NOCS_mask(image)
    h, w, c = image.shape

    # if mask.sum() < min_points:
    #     scale_factor = min_points / (mask.sum() + 1e-8)
    #     image_new_shape = (int(h * scale_factor), int(w * scale_factor))
    #     image = resize(image, image_new_shape, anti_aliasing=False)
    #     mask = resize(mask * 1.0, image_new_shape, anti_aliasing=False)
    mask = mask == 1.0
        
    pointcloud = image[mask, :]

    return pointcloud

if __name__ == "__main__":

    image_path = "/home/husky/Documents/overfit/a2d1b78e03f3cc39d1e95557cb698cdf/frame_00000001_NOXRayTL_00.png"

    image = read_NOCS_image(image_path)
    pcd = extract_NOCS_pointcloud(image)
    vis.visualize_pointclouds([pcd.T], [pcd.T])
