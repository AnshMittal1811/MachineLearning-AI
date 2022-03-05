#coding=utf-8
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import torchvision.transforms as transforms
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calMetric_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict==1)
    fn = np.sum(label == 1)
    return tp,fp+fn-tp

def nearest_interpolation(lr_feature_high, fake_image):
    scale = fake_image.size(2)//lr_feature_high.size(2)
    batch_size = lr_feature_high.size(0)
    channels = lr_feature_high.size(1)

    tmp_feature = fake_image
    for m in range(batch_size):
        for n in range(channels):
            new_lr_feature_high = lr_feature_high[m][n].unsqueeze(0).unsqueeze(0)
            a = torch.nn.functional.interpolate(new_lr_feature_high, scale_factor = scale,  mode='nearest', align_corners=None)
            tmp_feature[m][n] = a.squeeze()
    return tmp_feature


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result


def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class LoadDatasetFromFolder(Dataset):
    def __init__(self, args, hr1_path, lr2_path, hr2_path, lab_path):
        super(LoadDatasetFromFolder, self).__init__()
        datalist = [name for name in os.listdir(hr1_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.lr2_filenames = [join(lr2_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [join(lab_path, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize= False)
        self.label_transform = get_transform()

    def __getitem__(self, index):
        hr1_img = self.transform(Image.open(self.hr1_filenames[index]).convert('RGB'))
        lr2_img = self.transform(Image.open(self.lr2_filenames[index]).convert('RGB'))
        hr2_img = self.transform(Image.open(self.hr2_filenames[index]).convert('RGB'))

        label = self.label_transform(Image.open(self.lab_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        return hr1_img, lr2_img, hr2_img, label

    def __len__(self):
        return len(self.hr1_filenames)

class LoadDatasetFromFolder_CD(Dataset):
    def __init__(self, args, hr1_path, hr2_path, lab_path):
        super(LoadDatasetFromFolder_CD, self).__init__()
        datalist = [name for name in os.listdir(hr1_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [join(lab_path, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize= True)
        self.label_transform = get_transform()

    def __getitem__(self, index):
        hr1_img = self.transform(Image.open(self.hr1_filenames[index]).convert('RGB'))
        hr2_img = self.transform(Image.open(self.hr2_filenames[index]).convert('RGB'))

        label = self.label_transform(Image.open(self.lab_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        return hr1_img, hr2_img, label

    def __len__(self):
        return len(self.hr1_filenames)