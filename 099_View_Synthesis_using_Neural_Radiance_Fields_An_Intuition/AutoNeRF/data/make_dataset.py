
# Loads a original NeRF dataset and converts it to our format



from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import json
from math import tan


class AutoNeRF_Dataset(Dataset):
    def __init__(self, pth=None):
        self.pth = None
        self.images = None
        self.poses = None
        self.focal_length = None
        
        if pth is not None:
            self.pth = pth
            loaded = np.load(self.pth)

            self.images = loaded["images"]
            self.poses = loaded["poses"]
            self.focal_length = loaded["focal"]


    def __getitem__(self, index):
        image = self.images[index]
        pose = self.poses[index]

        return image, pose


    def __len__(self):
        if self.images is not None:
            return self.images.shape[0]
        return 0

    def save(self):
        np.savez_compressed(self.pth,
                            images = self.images,
                            poses = self.poses,
                            focal = self.focal_length
                            )


if __name__ == "__main__":
    image_pth = "/home/sircrashalot/Schreibtisch/chair/train/"
    save_pth = "/home/sircrashalot/Schreibtisch/chair/dataset/"
    transform_path = "/home/sircrashalot/Schreibtisch/chair/transforms_train.json"


    #n = 5
    size = (100,100)

    with open(transform_path, "r") as f:
        data = json.load(f)

    FOV_json = data["camera_angle_x"]
    tr_json = data["frames"]
    
    tr_matrices = []
    resized_images = []


    for i, frame in enumerate(tr_json):
        image_filename = frame["file_path"].split("/")[-1] + ".png"
        tr_matrix = np.array(frame["transform_matrix"], dtype=np.float32)
        tr_matrices.append(tr_matrix)

        im = Image.open(image_pth + image_filename)
        im.load()
        
        im2 = im.resize(size, Image.ANTIALIAS)
        #im_resize = im2.convert("RGB")
        im_resize = Image.new("RGB", size, (0,0,0))
        im_resize.paste(im2, mask=im2.split()[3])
        im_resize.save(save_pth + image_filename, "PNG", quality=100)

        arr = np.array(im_resize, dtype=np.float32)[None, ...]/255.
        resized_images.append(arr)
        
        #if i>n: break


    def get_focal_length(fov_x):
        """
        focal length = new_size/(tan(fov_x/2)*2)
        """
        focal_length = size[0]/ (tan(fov_x/2)*2)
        return np.array(focal_length, dtype=np.float32)

    def np_stack(x):
        z = np.dstack(x)
        z = z.transpose(2,0,1)
        return z

    tr_matrices = np_stack(tr_matrices)
    resized_images = np.vstack(resized_images)
    
    FOV = float(FOV_json)
    focal_length = get_focal_length(FOV)

    dataset = AutoNeRF_Dataset()
    dataset.pth = save_pth + "hotdog"
    dataset.images = resized_images
    dataset.poses = tr_matrices
    dataset.focal_length = focal_length

    print("DEBUG:")
    print("images: ", dataset.images.shape, dataset.images.dtype)
    print("poses: ", dataset.poses.shape, dataset.poses.dtype)
    print("focal length: ", dataset.focal_length)
    dataset.save()
    print("saved dataset to ", dataset.pth)


    #print(focal_length, type(focal_length))
    #print(tr_matrices)
    #tr_matrices = np.dstack(tr_matrices)
    #tr_matrices = tr_matrices.transpose(2, 0, 1)

    #resized_images =
