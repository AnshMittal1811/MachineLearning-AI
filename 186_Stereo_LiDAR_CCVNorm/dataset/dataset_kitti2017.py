"""
TorchDatset of KITTI Depth Completion dataset with stereo RGB pairs.

Dataset link: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion

Note:
- RGB normalize
- will transforms for depth affects disparity or depth?!
- fix size random crop
- no width 1226 focal length for disparity
"""

import os
import random
import numpy as np
from easydict import EasyDict
from skimage import io
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


EXCLUDED_SEQUENCES = [
    '2011_09_28_drive_0002_sync',
    '2011_09_29_drive_0004_sync',
    '2011_09_26_drive_0005_sync',
    '2011_09_26_drive_0009_sync',
    '2011_09_26_drive_0011_sync',
    '2011_09_26_drive_0013_sync',
    '2011_09_26_drive_0014_sync',
    '2011_09_26_drive_0015_sync',
    '2011_09_26_drive_0017_sync',
    '2011_09_26_drive_0018_sync',
    '2011_09_26_drive_0019_sync',
    '2011_09_26_drive_0022_sync',
    '2011_09_26_drive_0027_sync',
    '2011_09_26_drive_0028_sync',
    '2011_09_26_drive_0029_sync',
    '2011_09_26_drive_0032_sync',
    '2011_09_26_drive_0036_sync',
    '2011_09_26_drive_0046_sync',
    '2011_10_03_drive_0047_sync',
    '2011_09_26_drive_0051_sync',
    '2011_09_26_drive_0056_sync',
    '2011_09_26_drive_0057_sync',
    '2011_09_26_drive_0059_sync',
    '2011_09_26_drive_0070_sync',
    '2011_09_29_drive_0071_sync',
    '2011_09_26_drive_0084_sync',
    '2011_09_26_drive_0096_sync',
    '2011_09_26_drive_0101_sync',
    '2011_09_26_drive_0104_sync'
]


class DatasetKITTI2017(Dataset):
    FIXED_SHAPE = (256, 1216) # NOTE: Crop top; different from dataset_kitti2017

    def __init__(self, rgb_dir, depth_dir, mode, output_size, use_subset=False, to_disparity=False, fix_random_seed=False, exlude_data2015=False):
        # Check arguments
        assert mode in ['train', 'val', 'my_val', 'my_test'], 'Invalid mode for DatasetKITTI2017FullSize'
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.mode = mode
        self.output_size = output_size
        self.use_subset = use_subset
        self.to_disparity = to_disparity
        self.exlude_data2015 = exlude_data2015

        if fix_random_seed:
            random.seed(100)
            np.random.seed(seed=100)

        # Get all data path
        self.left_data_path, self.right_data_path = get_kitti2017_datapath(self.rgb_dir, self.depth_dir, self.mode, self.exlude_data2015)
        if self.use_subset is not False:
            assert isinstance(self.use_subset, int)
            random.seed(100) # NOTE: will affect other part of code?!
            subset_idx = random.sample(range(len(self.left_data_path['rgb'])), self.use_subset)
            get_subset = lambda x, s_i=subset_idx: [x_ for i_, x_ in enumerate(x) if i_ in s_i]
            self.left_data_path = dict(map(lambda kv, f=get_subset: (kv[0], f(kv[1])), self.left_data_path.items()))
            self.right_data_path = dict(map(lambda kv, f=get_subset: (kv[0], f(kv[1])), self.right_data_path.items()))

        # Define data transform
        self.transform = EasyDict()
        if self.mode in ['train']:
            self.transform.rgb = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
            self.transform.depth = transforms.Compose([
                transforms.ToPILImage(mode='F'), # NOTE: is this correct?!
                transforms.ToTensor()
            ])
        else: # val
            self.transform.rgb = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
            self.transform.depth = transforms.Compose([
                transforms.ToPILImage(mode='F'), # NOTE: is this correct?!
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        # Get data
        while(True):
            try: # NOTE: there are broken images in the dataset; skip those broken images
                left_rgb = read_rgb(self.left_data_path['rgb'][idx])
                img_h, img_w = left_rgb.shape[:2]
                while self.to_disparity and (img_w not in [1242, 1241, 1224, 1238, 1226]):
                    idx = random.randint(0, len(self.left_data_path['rgb']))
                    left_rgb = read_rgb(self.left_data_path['rgb'][idx])
                    img_h, img_w = left_rgb.shape[:2]
                right_rgb = read_rgb(self.right_data_path['rgb'][idx])
            except: # Encounter broken RGB
                idx = random.randint(0, len(self.left_data_path['rgb']))
                continue
            left_sdepth = read_depth(self.left_data_path['sdepth'][idx])
            left_depth = read_depth(self.left_data_path['depth'][idx])
            right_sdepth = read_depth(self.right_data_path['sdepth'][idx])
            right_depth = read_depth(self.right_data_path['depth'][idx])
            break

        # Convert depth to disparity. NOTE: Should be done before cropping and data transforms
        if self.to_disparity:
            # NOTE: scaling will affect disparity value
            #s = self.FIXED_SHAPE[1] / img_w # NOTE: BUGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
            s = 1

            left_sdisp = depth2disp(left_sdepth) / s
            left_disp = depth2disp(left_depth) / s
            right_sdisp = depth2disp(right_sdepth) / s
            right_disp = depth2disp(right_depth) / s

        # Crop to fixed size
        def crop_fn(x):
            start_h = img_h - self.FIXED_SHAPE[0]
            start_w = 0
            return x[start_h:start_h+self.FIXED_SHAPE[0], start_w:start_w+self.FIXED_SHAPE[1]]
        left_rgb, left_sdepth, left_depth = list(map(crop_fn, [left_rgb, left_sdepth, left_depth]))
        right_rgb, right_sdepth, right_depth = list(map(crop_fn, [right_rgb, right_sdepth, right_depth]))
        if self.to_disparity:
            left_sdisp, left_disp = list(map(crop_fn, [left_sdisp, left_disp]))
            right_sdisp, right_disp = list(map(crop_fn, [right_sdisp, right_disp]))
        if self.output_size[0] < self.FIXED_SHAPE[0] or self.output_size[1] < self.FIXED_SHAPE[1]:
            x1 = random.randint(0, self.FIXED_SHAPE[1]-self.output_size[1])
            y1 = random.randint(0, self.FIXED_SHAPE[0]-self.output_size[0])
            def rand_crop(x):
                return x[y1:y1+self.output_size[0], x1:x1+self.output_size[1]]
            left_rgb, left_sdepth, left_depth = list(map(rand_crop, [left_rgb, left_sdepth, left_depth]))
            right_rgb, right_sdepth, right_depth = list(map(rand_crop, [right_rgb, right_sdepth, right_depth]))
            if self.to_disparity:
                left_sdisp, left_disp = list(map(rand_crop, [left_sdisp, left_disp]))
                right_sdisp, right_disp = list(map(rand_crop, [right_sdisp, right_disp]))

        # Perform transforms
        data = dict()
        data['left_rgb'], data['right_rgb'] = list(map(self.transform.rgb, [left_rgb, right_rgb]))
        data['left_sd'], data['right_sd'] = list(map(self.transform.depth, [left_sdepth, right_sdepth]))
        data['left_d'], data['right_d'] = list(map(self.transform.depth, [left_depth, right_depth]))
        if self.to_disparity:
            data['left_sdisp'], data['right_sdisp'] = list(map(self.transform.depth, [left_sdisp, right_sdisp]))
            data['left_disp'], data['right_disp'] = list(map(self.transform.depth, [left_disp, right_disp]))
        data['width'] = img_w

        return data

    def __len__(self):
        return len(self.left_data_path['rgb'])


def depth2disp(depth):
    """ Convert depth to disparity for KITTI dataset.
        NOTE: depth must be the original rectified images.
        Ref: https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py """
    baseline = 0.54
    width_to_focal = dict()
    width_to_focal[1242] = 721.5377
    width_to_focal[1241] = 718.856
    width_to_focal[1224] = 707.0493
    width_to_focal[1226] = 708.2046 # NOTE: [wrong] assume linear to width 1224
    width_to_focal[1238] = 718.3351

    focal_length = width_to_focal[depth.shape[1]]
    invalid_mask = depth <= 0
    disp = baseline * focal_length / (depth + 1E-8)
    disp[invalid_mask] = 0
    return disp


def read_rgb(path):
    """ Read raw RGB and DO NOT perform any process to the image """
    rgb = io.imread(path)
    return rgb


def read_depth(path):
    """ Depth maps (annotated and raw Velodyne scans) are saved as uint16 PNG images,
        which can be opened with either MATLAB, libpng++ or the latest version of
        Python's pillow (from PIL import Image). A 0 value indicates an invalid pixel
        (ie, no ground truth exists, or the estimation algorithm didn't produce an
        estimate for that pixel). Otherwise, the depth for a pixel can be computed
        in meters by converting the uint16 value to float and dividing it by 256.0:

        disp(u,v)  = ((float)I(u,v))/256.0;
        valid(u,v) = I(u,v)>0;
    """
    depth = Image.open(path)
    depth = np.array(depth).astype(np.float32) / 256.0
    return depth[:, :, np.newaxis]
    

def get_kitti2017_datapath(rgb_dir, depth_dir, mode, exlude_data2015=False):
    """ Read path to all data from KITTI Depth Completion dataset """
    left_data_path = {'rgb': [], 'sdepth': [], 'depth': []}
    right_data_path = {'rgb': [], 'sdepth': [], 'depth': []}
    dir_name_list = sorted(os.listdir(os.path.join(depth_dir, mode)))
    for dir_name in dir_name_list:
        if exlude_data2015 and dir_name in EXCLUDED_SEQUENCES: # skip some sequences overlapped with KITTI2015
            continue
        # Directory of RGB images
        rgb_left_dir = os.path.join(rgb_dir, dir_name[:-16], dir_name, 'image_02', 'data')
        rgb_right_dir = os.path.join(rgb_dir, dir_name[:-16], dir_name, 'image_03', 'data')
        # Directory of ground truth depth maps
        depth_left_dir = os.path.join(depth_dir, mode, dir_name, 'proj_depth', 'groundtruth', 'image_02')
        depth_right_dir = os.path.join(depth_dir, mode, dir_name, 'proj_depth', 'groundtruth', 'image_03')
        # Directory of sparse depth maps from LiDAR
        sdepth_left_dir = os.path.join(depth_dir, mode, dir_name, 'proj_depth', 'velodyne_raw', 'image_02')
        sdepth_right_dir = os.path.join(depth_dir, mode, dir_name, 'proj_depth', 'velodyne_raw', 'image_03')

        # Get image names (DO NOT obtain from raw data directory since the annotated data is pruned)
        file_name_list = sorted(os.listdir(depth_left_dir))

        for file_name in file_name_list:
            # Path to RGB images
            rgb_left_path = os.path.join(rgb_left_dir, file_name)
            rgb_right_path = os.path.join(rgb_right_dir, file_name)
            # Path to ground truth depth maps
            depth_left_path = os.path.join(depth_left_dir, file_name)
            depth_right_path = os.path.join(depth_right_dir, file_name)
            # Path to sparse depth maps
            sdepth_left_path = os.path.join(sdepth_left_dir, file_name)
            sdepth_right_path = os.path.join(sdepth_right_dir, file_name)

            # Add to list
            left_data_path['rgb'].append(rgb_left_path)
            left_data_path['sdepth'].append(sdepth_left_path)
            left_data_path['depth'].append(depth_left_path)
            right_data_path['rgb'].append(rgb_right_path)
            right_data_path['sdepth'].append(sdepth_right_path)
            right_data_path['depth'].append(depth_right_path)

    return left_data_path, right_data_path


def test_basic():
    """ Examine the correctness of the dataset class """
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import pdb
    
    # Setup dataset
    dataset = DatasetKITTI2017(rgb_dir='../data/kitti2017/rgb',
                               depth_dir='../data/kitti2017/depth',
                               mode='train',
                               output_size=(256, 512),
                               use_subset=False,
                               to_disparity=True)

    # Check data
    visualize_rgb = True
    visualize_sd = True
    visualize_d = True
    print('Dataset size = {}'.format(len(dataset)))
    for i, data in enumerate(dataset):
        # Unpack data
        data = EasyDict(data)
        left_rgb_np = data.left_rgb.numpy().transpose(1, 2, 0)
        right_rgb_np = data.right_rgb.numpy().transpose(1, 2, 0)
        left_sd_np = data.left_sd.numpy()[0]
        right_sd_np = data.left_sd.numpy()[0]
        left_d_np = data.left_d.numpy()[0]
        right_d_np = data.left_d.numpy()[0]

        # Visualization
        if visualize_rgb:
            fig, axes = plt.subplots(2, 1)
            axes[0].set_title('RGB (left)')
            axes[0].imshow(left_rgb_np)
            axes[1].set_title('RGB (right)')
            axes[1].imshow(right_rgb_np)
        if visualize_sd:
            fig, axes = plt.subplots(2, 1)
            axes[0].set_title('LiDAR (left)')
            axes[0].imshow(left_sd_np)
            axes[1].set_title('LiDAR (right)')
            axes[1].imshow(right_sd_np)
        if visualize_d:
            fig, axes = plt.subplots(2, 1)
            axes[0].set_title('Depth (left)')
            axes[0].imshow(left_d_np)
            axes[1].set_title('Depth (right)')
            axes[1].imshow(right_d_np)
        plt.show(block=False)
        pdb.set_trace()


if __name__ == '__main__':
    test_basic()
