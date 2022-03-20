# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""Code for getting the data loaders."""

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from util.lmdb_datasets import LMDBDataset
from thirdparty.lsun import LSUN, LSUNClass
import os
from torch._utils import _accumulate


class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """

    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_loaders(args):
    """Get data loaders for required dataset."""
    if args.data_location is None:
        args.data_location = os.path.join(args.root, 'data')
    return get_loaders_eval(args.dataset, args.data_location, args.distributed, args.training_batch_size, args.testing_batch_size)


def get_loaders_eval(dataset, root, distributed, training_batch_size, testing_batch_size, augment=True, drop_last_train=True, shuffle_train=True):
    if dataset == 'cifar10':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_cifar10()
        train_transform = train_transform if augment else valid_transform
        train_data = dset.CIFAR10(
            root=root, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(
            root=root, train=False, download=True, transform=valid_transform)
    elif dataset.startswith('celeba'):
        if dataset == 'celeba_64':
            resize = 64
            num_classes = 40
            train_transform, valid_transform = _data_transforms_celeba64(
                resize)
            train_transform = train_transform if augment else valid_transform
            train_data = LMDBDataset(
                root=root, name='celeba64', train=True, transform=train_transform, is_encoded=True)
            valid_data = LMDBDataset(
                root=root, name='celeba64', train=False, transform=valid_transform, is_encoded=True)
        elif dataset in {'celeba_256'}:
            num_classes = 1
            resize = int(dataset.split('_')[1])
            train_transform, valid_transform = _data_transforms_generic(resize)
            train_transform = train_transform if augment else valid_transform
            train_data = LMDBDataset(
                root=root, name='celeba', train=True, transform=train_transform)
            valid_data = LMDBDataset(
                root=root, name='celeba', train=False, transform=valid_transform)
        else:
            raise NotImplementedError
    elif dataset.startswith('imagenet'):
        num_classes = 1
        resize = int(dataset.split('_')[1])
        assert root.replace(
            '/', '')[-3:] == dataset.replace('/', '')[-3:], 'the size should match'
        train_transform, valid_transform = _data_transforms_generic(resize)
        train_data = LMDBDataset(
            root=root, name='imagenet-oord', train=True, transform=train_transform)
        valid_data = LMDBDataset(
            root=root, name='imagenet-oord', train=False, transform=valid_transform)
    elif dataset.startswith('lsun'):
        if dataset.startswith('lsun_bedroom'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_transform = train_transform if augment else valid_transform
            train_data = LSUN(root=root, classes=[
                              'bedroom_train'], transform=train_transform)
            valid_data = LSUN(root=root, classes=[
                              'bedroom_val'], transform=valid_transform)
        elif dataset.startswith('lsun_church'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_transform = train_transform if augment else valid_transform
            train_data = LSUN(root=root, classes=[
                              'church_outdoor_train'], transform=train_transform)
            valid_data = LSUN(root=root, classes=[
                              'church_outdoor_val'], transform=valid_transform)
        elif dataset.startswith('lsun_tower'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_transform = train_transform if augment else valid_transform
            train_data = LSUN(root=root, classes=[
                              'tower_train'], transform=train_transform)
            valid_data = LSUN(root=root, classes=[
                              'tower_val'], transform=valid_transform)
        elif dataset.startswith('lsun_cat'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_transform = train_transform if augment else valid_transform
            data = LSUNClass(root=root + '/cat', transform=train_transform)
            total_examples = len(data)
            train_size = int(0.9 * total_examples)   # use 90% for training
            train_data, valid_data = random_split_dataset(
                data, [train_size, total_examples - train_size])
        else:
            raise NotImplementedError
    elif dataset.startswith('imagenet'):
        num_classes = 1
        resize = int(dataset.split('_')[1])
        assert root.replace(
            '/', '')[-3:] == dataset.replace('/', '')[-3:], 'the size should match'
        train_transform, valid_transform = _data_transforms_generic(resize)
        train_transform = train_transform if augment else valid_transform
        train_data = LMDBDataset(
            root=root, name='imagenet-oord', train=True, transform=train_transform)
        valid_data = LMDBDataset(
            root=root, name='imagenet-oord', train=False, transform=valid_transform)
    elif dataset.startswith('ffhq'):
        num_classes = 1
        resize = 256
        train_transform, valid_transform = _data_transforms_generic(resize)
        train_transform = train_transform if augment else valid_transform
        train_data = LMDBDataset(
            root=root, name='ffhq', train=True, transform=train_transform)
        valid_data = LMDBDataset(
            root=root, name='ffhq', train=False, transform=valid_transform)
    else:
        raise NotImplementedError

    train_sampler, valid_sampler = None, None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_data)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=training_batch_size,
        shuffle=(train_sampler is None) and shuffle_train,
        sampler=train_sampler, pin_memory=True, num_workers=8, drop_last=drop_last_train)  # 8

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=testing_batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=1, drop_last=False)  # 1

    return train_queue, valid_queue, num_classes


def random_split_dataset(dataset, lengths, seed=0):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!")
    g = torch.Generator()
    g.manual_seed(seed)

    indices = torch.randperm(sum(lengths), generator=g)
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)]


def _data_transforms_cifar10():
    """Get data transforms for cifar10."""

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return train_transform, valid_transform


def _data_transforms_generic(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_celeba64(size):
    train_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_lsun(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from util.utils import tile_image
    train_queue, valid_queue, num_classes = get_loaders_eval('lsun_cat_256', root='/data1/datasets/LSUN',
                                                             distributed=False, training_batch_size=16, testing_batch_size=16, binarize_binary_datasets=False)
    print(len(train_queue), len(valid_queue))
    step = 0
    for b in valid_queue:
        print(b[0].shape)
        b = tile_image(b[0], n=4).permute(1, 2, 0).cpu().numpy()
        print(np.min(b), np.max(b))
        plt.imshow(b)
        plt.show()
        step += 1
        break
