"""
TorchDatset of KITTI 2015 Stereo dataset with stereo RGB pairs.

Dataset link: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
"""
import os
import random
import numpy as np
from easydict import EasyDict
from skimage import io
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class DatasetKITTI2015(Dataset):
    FIXED_SHAPE = (352, 1216)

    def __init__(self, root_dir, mode, output_size, random_sampling=None, fix_random_seed=False):
        # Check arguments
        assert mode in ['training'], 'Invalid mode for DatasetKITTI2015'
        self.root_dir = root_dir
        self.mode = mode
        self.output_size = output_size
        if random_sampling is None:
            self.sampler = None
        elif isinstance(random_sampling, float):
            self.sampler = UniformSamplerByPercentage(random_sampling)
        else:
            raise ValueError

        if fix_random_seed:
            random.seed(100)
            np.random.seed(seed=100)

        # Get all data path
        self.left_data_path, self.right_data_path = get_kitti2015_datapath(self.root_dir, self.mode, self.sampler)

        # Define data transform
        self.transform = EasyDict()
        if self.mode in ['training']:
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
                right_rgb = read_rgb(self.right_data_path['rgb'][idx])
            except: # Encounter broken RGB
                idx = random.randint(0, len(self.left_data_path['rgb']))
                continue
            left_disp = read_depth(self.left_data_path['disp'][idx])
            right_disp = read_depth(self.right_data_path['disp'][idx])
            if self.sampler is None:
                left_sdisp = depth2disp(read_depth(self.left_data_path['sdepth'][idx]))
                right_sdisp = depth2disp(read_depth(self.right_data_path['sdepth'][idx]))
            else:
                left_sdisp = self.sampler.sample(left_disp)
                right_sdisp = self.sampler.sample(right_disp)
            break

        # Crop to fixed size
        def crop_fn(x):
            start_h = img_h - self.FIXED_SHAPE[0] if img_h > self.FIXED_SHAPE[0] else 0
            start_w = 0
            end_w = min(img_w, start_w+self.FIXED_SHAPE[1])
            return x[start_h:start_h+self.FIXED_SHAPE[0], start_w:end_w]
        left_rgb, left_sdisp, left_disp = list(map(crop_fn, [left_rgb, left_sdisp, left_disp]))
        right_rgb, right_sdisp, right_disp = list(map(crop_fn, [right_rgb, right_sdisp, right_disp]))
        if self.output_size[0] < self.FIXED_SHAPE[0] or self.output_size[1] < self.FIXED_SHAPE[1]:
            x1 = random.randint(0, self.FIXED_SHAPE[1]-self.output_size[1])
            y1 = random.randint(0, self.FIXED_SHAPE[0]-self.output_size[0])
            def rand_crop(x):
                return x[y1:y1+self.output_size[0], x1:x1+self.output_size[1]]
            left_rgb, left_sdisp, left_disp = list(map(rand_crop, [left_rgb, left_sdisp, left_disp]))
            right_rgb, right_sdisp, right_disp = list(map(rand_crop, [right_rgb, right_sdisp, right_disp]))

        # Perform transforms
        data = dict()
        data['left_rgb'], data['right_rgb'] = list(map(self.transform.rgb, [left_rgb, right_rgb]))
        data['left_sdisp'], data['right_sdisp'] = list(map(self.transform.depth, [left_sdisp, right_sdisp]))
        data['left_disp'], data['right_disp'] = list(map(self.transform.depth, [left_disp, right_disp]))
        data['width'] = img_w

        return data

    def __len__(self):
        return len(self.left_data_path['rgb'])


class UniformSamplerByPercentage(object):
    """ (Numpy) Uniform sampling by number of sparse points """
    def __init__(self, percent_samples):
        super(UniformSamplerByPercentage, self).__init__()
        self.percent_samples = percent_samples
        self.max_depth = 100

    def sample(self, x):
        s_x = np.zeros_like(x)
        if self.max_depth is np.inf:
            prob = float(self.n_samples) / x.size
            mask_keep = np.random.uniform(0, 1, x.shape) < prob
            s_x[mask_keep] = x[mask_keep]
        else:
            sparse_mask = (x <= self.max_depth) & (x > 0)
            n_keep = sparse_mask.astype(np.float).sum()
            if n_keep == 0:
                raise ValueError('`max_depth` filter out all valid depth points')
            else:
                mask_keep = np.random.uniform(0, 1, x[sparse_mask].shape) < self.percent_samples
                tmp = np.zeros(mask_keep.shape)
                tmp[mask_keep] = x[sparse_mask][mask_keep]
                s_x[sparse_mask] = tmp

        return s_x


def depth2disp(depth):
    """ Convert depth to disparity for KITTI dataset.
        NOTE: depth must be the original rectified images.
        Ref:  """
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
 

def get_kitti2015_datapath(root_dir, mode, sampler=None):
    """ Read path to all data from KITTI Stereo 2015 dataset """
    left_data_path = {'rgb': [], 'sdepth': [], 'disp': [], 'disp_occ': []}
    right_data_path = {'rgb': [], 'sdepth': [], 'disp': [], 'disp_occ': []}
    if sampler is None:
        fname_list = sorted(os.listdir(os.path.join(root_dir, mode, 'velodyne_2')))
        fname_list = [f[:-4]+'_10.png' for f in fname_list]
    else:
        fname_list = sorted(os.listdir(os.path.join(root_dir, mode, 'image_2')))
        fname_list = [f for f in fname_list if f[-6:]=='10.png']
    for fname in fname_list:
        left_data_path['rgb'].append(os.path.join(root_dir, mode, 'image_2', fname))
        right_data_path['rgb'].append(os.path.join(root_dir, mode, 'image_3', fname))
        left_data_path['disp'].append(os.path.join(root_dir, mode, 'disp_noc_0', fname))
        right_data_path['disp'].append(os.path.join(root_dir, mode, 'disp_noc_1', fname))
        left_data_path['disp_occ'].append(os.path.join(root_dir, mode, 'disp_occ_0', fname))
        right_data_path['disp_occ'].append(os.path.join(root_dir, mode, 'disp_occ_1', fname))
        if sampler is None:
            left_data_path['sdepth'].append(os.path.join(root_dir, mode, 'velodyne_2', fname[:-7]+'.png'))
            right_data_path['sdepth'].append(os.path.join(root_dir, mode, 'velodyne_3', fname[:-7]+'.png'))
    return left_data_path, right_data_path


def test_basic():
    """ Examine the correctness of the dataset class """
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import pdb
    
    # Setup dataset
    dataset = DatasetKITTI2015(root_dir='../data/kitti_stereo/data_scene_flow',
                               mode='training',
                               output_size=(352, 1224),
                               random_sampling=0.15)

    # Check data
    visualize_rgb = True
    visualize_sd = False
    visualize_d = False
    print('Dataset size = {}'.format(len(dataset)))
    for i, data in enumerate(dataset):
        # Unpack data
        data = EasyDict(data)
        left_rgb_np = data.left_rgb.numpy().transpose(1, 2, 0)
        right_rgb_np = data.right_rgb.numpy().transpose(1, 2, 0)
        left_sd_np = data.left_sdisp.numpy()[0]
        right_sd_np = data.left_sdisp.numpy()[0]
        left_d_np = data.left_disp.numpy()[0]
        right_d_np = data.left_disp.numpy()[0]
        #print(i, left_rgb_np.shape)
        #import pdb; pdb.set_trace()

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
