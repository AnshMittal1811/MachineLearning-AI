import torch
from torch.utils.data import Dataset
import numpy as np
import os
import os.path
from PIL import Image
import pandas as pd
from torchvision import transforms
from tools.utils import nested_tensor_from_tensor_list


def read_data(file):
    info = pd.read_csv(file)
    img_list = info['file_path'].tolist()
    label_list = info['label'].tolist()
    return img_list, label_list

def make_dataset(csv_file, root=None):
    dataset = []

    imgs, labels = read_data(csv_file)

    for i in range(len(imgs)):
        if root:
            imgs[i] = os.path.join(root, imgs[i]) # from relative path to absolute path
        dataset.append((imgs[i], labels[i]))

    return dataset


def create_train_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                                0.8, 1.5], saturation=[0.2, 1.5]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def create_val_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class SeqDeepFakeDataset(Dataset):

    def __init__(self,
                 cfg=None,
                 data_root=None,
                 mode="train",
                 dataset_name=None
                 ):
        super().__init__()
        self.mode = mode
        self.cfg = cfg
        if self.mode == "train":
            self.transforms = create_train_transforms(cfg.imgsize)
        elif self.mode in ["val", "test"]:
            self.transforms = create_val_transforms(cfg.imgsize)
        else:
            raise ValueError(f"WRONG INPUT MODE: {self.mode}!")

        self.data = make_dataset(os.path.join(data_root, f"{dataset_name}/annotations/{mode}.csv"), root=data_root)
        
        self.SOS_token_id = cfg.SOS_token_id
        self.EOS_token_id = cfg.EOS_token_id
        self.PAD_token_id = cfg.PAD_token_id

    def __getitem__(self, index: int):
        img_path, label = self.data[index]
        if self.mode in ["train"]:
            label_list_split = label.split(',')

            label_list = [int(label_list_split[0][1]), int(label_list_split[1][1]), int(label_list_split[2][1]), int(label_list_split[3][1]), int(label_list_split[4][1])]

            caption = np.array(label_list)

            if np.count_nonzero(caption) == 0:
                caption_mask = (1-np.array([1, 1, 0, 0, 0, 0, 0])).astype(bool)
                caption = np.insert(caption, [0, 0], [self.SOS_token_id, self.EOS_token_id])
                caption[np.where(caption_mask==True)] = self.PAD_token_id

            elif np.count_nonzero(caption) == len(caption):
                caption_mask = (1-np.array([1, 1, 1, 1, 1, 1, 1])).astype(bool)
                caption = np.insert(caption, [0, 5], [self.SOS_token_id, self.EOS_token_id])
            else:
                first_zero_idx = np.where(caption==0)[0][0]
                caption = np.insert(caption, [0, first_zero_idx], [self.SOS_token_id, self.EOS_token_id])
                EOS_idx = np.where(caption==self.EOS_token_id)[0][0]
                caption_mask = (1-np.pad(np.ones(EOS_idx+1), (0, len(caption)-(EOS_idx+1)))).astype(bool)
                caption[np.where(caption_mask==True)] = self.PAD_token_id
            
            image = Image.open(img_path).convert('RGB')
            if self.transforms:
                image = self.transforms(image)

            image = nested_tensor_from_tensor_list(self.cfg.imgsize, image.unsqueeze(0))

            return image.tensors.squeeze(0), image.mask.squeeze(0), caption, caption_mask
        elif self.mode in ["val", "test"]:
            label_list_split = label.split(',')
            label_list = [int(label_list_split[0][1]), int(label_list_split[1][1]), int(label_list_split[2][1]), int(label_list_split[3][1]), int(label_list_split[4][1])]
            
            image = Image.open(img_path).convert('RGB')
            if self.transforms:
                image = self.transforms(image)

            return image, torch.FloatTensor(label_list)
        else:
            raise ValueError(f"WRONG INPUT MODE: {self.mode}!")

    def __len__(self):
        return len(self.data)
