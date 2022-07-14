import os
from torch.utils.data import Dataset
from PIL import Image
from util.camera_transformations import *
import re
import torchvision

from abc import ABC, abstractmethod


class ToNumpy(object):
    def __call__(self, sample):
        return np.array(sample)


class ClipDepth(object):
    '''Set maximal depth'''

    def __init__(self, maxDepth):
        self.maxDepth = maxDepth

    def __call__(self, sample):
        sample[sample > self.maxDepth] = self.maxDepth
        return sample


class DiskDataset(Dataset, ABC):
    '''
    Loads samples from files satisfying the following directory structure:
    TODO describe
    '''

    # regex to read lines from the camera .txt file
    cam_pattern = "(?P<id>.*\w).*= \[(?P<x>.*), (?P<y>.*), (?P<z>.*)\].*"

    def __init__(self,
                 path,
                 maxDepth,
                 imageInputShape,
                 sampleOutput=True,
                 input_as_segmentation=False,
                 inverse_depth=False,
                 cacheItems=False,
                 transform=None):
        '''

        :param path: path/to/<base>/files. Needs to be a directory with .png, .depth and .txt files
        :param maxDepth: maximum depth that is accepted. Everything above that will be cut to given value.
        :param imageInputShape: the original image input shape for the dataset that gets used. Necessary to reshape depth values into correct array dimensions.
        :param sampleOutput: whether or not to sample an output image. Implementations specify how the output is retrieved.
        :param inverse_depth: If true, depth.pow(-1) is returned for the depth file (changing depth BEFORE applying transform object).
        :param cacheItems: If true, all items will be stored in RAM dictionary, once they were accessed.
        :param transform: transform that should be applied to the input image AND the target depth
        '''

        # SAVE INPUT ARGUMENTS
        self.imageInputShape = imageInputShape
        self.inverse_depth = inverse_depth
        self.maxDepth = maxDepth
        self.path = path
        self.sampleOutput = sampleOutput
        self.cacheItems = cacheItems
        self.input_as_segmentation = input_as_segmentation

        # SAVE TRANSFORM OBJECT IN CLASS
        self.transform = transform
        # Fix for this issue: https://github.com/pytorch/vision/issues/2194
        if self.transform and isinstance(self.transform.transforms[-1], torchvision.transforms.ToTensor):
            self.transform_depth = torchvision.transforms.Compose([
                *self.transform.transforms[:-1],
                ToNumpy(),
                ClipDepth(maxDepth),
                torchvision.transforms.ToTensor()
            ])
        else:
            self.transform_depth = self.transform

        # LOAD K
        self.K, self.Kinv = self.load_int_cam()

        # LOAD DATA
        dir_content = os.listdir(path)
        self.img_rgb, \
        self.img_seg, \
        self.depth, \
        self.has_depth, \
        self.depth_binary, \
        self.has_binary_depth, \
        self.cam, \
        self.size, \
        self.dynamics, \
        self.moved_rgb_gt_for_evaluation_only = self.load_data(dir_content)

        # CREATE OUTPUT PAIR
        if self.sampleOutput:
            self.inputToOutputIndex = self.create_input_to_output_sample_map()

        # SETUP EMPTY CACHE
        self.itemCache = [None for i in range(self.size)]

    def load_image(self, idx, load_seg_image=False):
        if load_seg_image:
            seg_image = Image.open(os.path.join(self.path, self.img_seg[idx]))
            if seg_image.mode == "RGBA":
                seg_image = seg_image.convert("RGB")
            return seg_image
        else:
            return Image.open(os.path.join(self.path, self.img_rgb[idx]))

    def load_moved_gt_rgb_image_for_evaluation(self, idx):
        if self.moved_rgb_gt_for_evaluation_only is not None:
            return Image.open(os.path.join(self.path, self.moved_rgb_gt_for_evaluation_only[idx]))
        else:
            return None

    def load_depth(self, idx):
        if self.has_binary_depth:
            # read faster from .depth.npy file - this is much faster than parsing the char-based .depth file from ICL directly.
            depth = np.load(os.path.join(self.path, self.depth_binary[idx]))
        elif self.has_depth:
            with open(os.path.join(self.path, self.depth[idx])) as f:
                depth = [float(i) for i in f.read().split(' ') if i.strip()]  # read .depth file
                depth = np.asarray(depth, dtype=np.float32).reshape(
                    self.imageInputShape)  # convert to same format as image HxW
        else:
            return None

        # Implementation specific
        depth = self.modify_depth(depth)

        # invert depth
        if self.inverse_depth:
            depth = np.power(depth, -1)

        return Image.fromarray(depth, mode='F')  # return as float PIL Image

    def load_ext_cam(self, idx):
        cam = {}  # load the .txt file in this dict
        with open(os.path.join(self.path, self.cam[idx])) as f:
            for line in f:
                m = re.match(DiskDataset.cam_pattern,
                             line)  # will match everything except angle, but that is not needed anyway
                if m is not None:
                    cam[m["id"]] = np.zeros(3)
                    cam[m["id"]][0] = float(m["x"])
                    cam[m["id"]][1] = float(m["y"])
                    cam[m["id"]][2] = float(m["z"])

        # calculate RT matrix, taken from: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/computeRT.m
        z = cam["cam_dir"] / np.linalg.norm(cam["cam_dir"])
        x = np.cross(cam["cam_up"], z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        # combine in correct shape
        RT = np.column_stack((x, y, z, cam["cam_pos"]))
        RT = np.vstack([RT, [0, 0, 0, 1]])
        RT = RT.astype(np.float32)

        # calculate RTinv from RT
        RTinv = np.linalg.inv(RT).astype(np.float32)

        # return as torch tensor
        RT = torch.from_numpy(RT)
        RTinv = torch.from_numpy(RTinv)

        return RT, RTinv

    def load_dynamics(self, input_image, output_image=None):
        # load transformation
        transformation = self.dynamics["transformation"]
        transformation = np.asarray(transformation).reshape((3, 4))
        transformation = np.vstack([transformation, [0, 0, 0, 1]])
        transformation = self.modify_dynamics_transformation(transformation).astype(np.float32)

        # load color of moved object in segmentation images (input and output have same color for that object)
        color = self.dynamics["color"]
        color = np.asarray(color)

        # convert input and output images
        input_image = np.asarray(input_image) / 255.0
        if output_image is not None:
            output_image = np.asarray(output_image) / 255.0

        # calculate input mask
        input_mask = np.isclose(input_image, color)
        input_mask = input_mask[:, :, 0] & input_mask[:, :, 1] & input_mask[:, :, 2]
        input_mask = Image.fromarray(input_mask)

        # calculate output mask
        if output_image is not None:
            output_mask = np.isclose(output_image, color)
            output_mask = output_mask[:, :, 0] & output_mask[:, :, 1] & output_mask[:, :, 2]
            output_mask = Image.fromarray(output_mask)

        # apply transformation to masks
        if self.transform:
            input_mask = self.transform(input_mask).bool()
            if output_image is not None:
                output_mask = self.transform(output_mask).bool()
            if isinstance(self.transform.transforms[-1], torchvision.transforms.ToTensor):
                transformation = torchvision.transforms.ToTensor()(transformation)

        result = {
            "transformation": transformation,
            "input_mask": input_mask,
        }

        if output_image is not None:
            result["output_mask"] = output_mask

        return result

    @abstractmethod
    def load_int_cam(self):
        """
        Calculates the K and Kinv matrix for the dataset.
        :return: K, Kinv
        """
        pass

    @abstractmethod
    def modify_depth(self, depth):
        """
        Calculates modifications necessary for the concrete dataset after reading depth from file.
        :param depth: depth as read from file
        :return: depth changed as needed
        """
        pass

    # @abstractmethod
    def modify_dynamics_transformation(self, transformation):
        """
        Calculates modifications necessary for concrete dataset after reading transformation from file.
        :param transformation:  transformation as read from file with [0 0 0 1] added as last row, so it is a 4x4 RT matrix
        :return: transformation changed as needed
        """
        raise NotImplementedError()

    @abstractmethod
    def load_data(self, dir_content):
        """
        Each dataset defines how to load the data:
            - img_rgb: list of all paths to images
            - img_seg: list of all paths to seg images
            - depth: list of all paths to depth files in .depth format
            - has_depth: if depth list is empty or not
            - depth_binary: list of all paths to depth files in .depth.npy format
            - has_binary_depth: if depth_binary list is empty or not
            - cam: list of all paths to extrinsic camera .txt files
            - size: how many data samples are available
            - dynamics: list of path to the .dynamics file. If list is None, then we do not have dynamics for this dataset available.
            - moved_rgb_gt_for_evaluation_only: list of moved rgb images only used for evaluation metrics (never during training)

        :param dir_content: list of all files in the root path (self.path)

        :return: tuple (img_rgb, img_seg, depth, has_depth, depth_binary, has_binary_depth, cam, size, dynamics, moved_rgb_gt_for_evaluation_only)
        """
        pass

    @abstractmethod
    def create_input_to_output_sample_map(self):
        """
        Each dataset decides how to associate an img in self.img to an output image in self.img.

        :return: list of length self.img where the i-th entry is an index of self.img specifying the output image.
        """
        pass

    def __getitem__(self, idx):
        """

        :param idx: item to choose
        :return: dictionary with following format:
            {
                'image': image,
                'depth': depth,
                'cam': cam,
                'output': output,
                'dynamics': dynamics
            }
            where
              image is a WxHxC matrix of floats,
              depth is a WxH matrix of floats,
              cam is a dictionary:
                {
                    'RT1': RT1,
                    'RT1inv': RT1inv,
                    'RT2': RT2,
                    'RT2inv': RT2inv,
                    'K': ICLNUIMDataset.K,
                    'Kinv': ICLNUIMDataset.Kinv
                }
                where
                  RT1 is a 4x4 extrinsic matrix of the idx-th item,
                  RT2 is a 4x4 extrinsic matrix of a random neighboring item or None (see self.sampleOutput)
                  K is a 4x4 intrinsic matrix (constant over all items) with 4th row/col added for convenience,
                  *inv is the inverted matrix
              output is a dictionary or None (see self.sampleOutput):
                {
                  'image': output_image,
                  'idx': output_idx
                }
                where
                  image is a random neighboring image
                  idx is the index of the neighboring image (and of cam['RT2'])
              dynamics is a dictionary or None:
                {
                    "mask": pixel mask of input image that is 1 for all pixels that were moved and 0 for all other pixels.
                    "transformation": 4x4 [R|T] matrix that was applied to the object identified by the pixel mask as the dynamic change.
                }



        """

        # LOAD FROM CACHE
        if self.itemCache[idx] is not None:
            return self.itemCache[idx]

        # LOAD INPUT IMAGE
        image = self.load_image(idx, self.input_as_segmentation)

        # LOAD INPUT SEG IMAGE
        if self.img_seg is not None:
            seg_image = self.load_image(idx, True)
        else:
            seg_image = None

        # LOAD INPUT DEPTH
        depth = self.load_depth(idx)

        # LOAD INPUT CAM
        RT1, RT1inv = self.load_ext_cam(idx)
        cam = {
            'RT1': RT1,
            'RT1inv': RT1inv,
            'K': self.K,
            'Kinv': self.Kinv
        }

        # LOAD OUTPUT
        output = None
        if self.sampleOutput:
            # lookup output for this sample
            output_idx = self.inputToOutputIndex[idx]

            # load image and depth of new index
            output_image_rgb = self.load_image(output_idx, False)
            if self.img_seg is not None:
                output_image_seg = self.load_image(output_idx, True)
            else:
                output_image_seg = None
            output_depth = self.load_depth(output_idx)

            # load moved_gt_rgb_image_for_evaluation_only if available
            moved_output_gt_rgb_image_for_evaluation_only = self.load_moved_gt_rgb_image_for_evaluation(output_idx)

            # load cam of new index
            RT2, RT2inv = self.load_ext_cam(output_idx)

            cam['RT2'] = RT2
            cam['RT2inv'] = RT2inv

            output = {
                'image': output_image_rgb,
                'seg': output_image_seg,
                'depth': output_depth,
                'idx': output_idx,
                'gt_moved_rgb_for_evaluation_only': moved_output_gt_rgb_image_for_evaluation_only
            }

        # LOAD DYNAMICS
        if self.dynamics is not None:
            if self.sampleOutput:
                dynamics = self.load_dynamics(seg_image, output_image_seg)
            else:
                dynamics = self.load_dynamics(seg_image, None)
        else:
            dynamics = {}

        # CONSTRUCT SAMPLE
        sample = {
            'image': image,
            'seg': seg_image,
            'depth': depth,
            'cam': cam,
            'output': output,
            'dynamics': dynamics
        }

        # APPLY TRANSFORM OBJECT
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            if sample['seg'] is not None:
                sample['seg'] = self.transform(sample['seg'])
            if sample['depth'] is not None:
                sample['depth'] = self.transform_depth(sample['depth'])

            if self.sampleOutput:
                sample['output']['image'] = self.transform(sample['output']['image'])
                if sample['output']['seg'] is not None:
                    sample['output']['seg'] = self.transform(sample['output']['seg'])
                if sample['output']['depth'] is not None:
                    sample['output']['depth'] = self.transform_depth(sample['output']['depth'])
                if sample['output']['gt_moved_rgb_for_evaluation_only'] is not None:
                    sample['output']['gt_moved_rgb_for_evaluation_only'] = self.transform(sample['output']['gt_moved_rgb_for_evaluation_only'])

        # STORE IN CACHE
        if self.cacheItems:
            self.itemCache[idx] = sample

        return sample

    def __len__(self):
        return self.size
