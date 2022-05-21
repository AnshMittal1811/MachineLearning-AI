import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


class DataAugmentationDINO:
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            A.ToGray(p=0.2)
        ])
        normalize = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

        # first global crop
        self.global_crop1 = A.Compose([
            A.RandomResizedCrop(224, 224, scale=global_crops_scale, interpolation=cv2.INTER_CUBIC),
            flip_and_color_jitter,
            A.GaussianBlur(p=1.0),
            normalize,
        ])
        # second global crop
        self.global_crop2 = A.Compose([
            A.RandomResizedCrop(224, 224, scale=global_crops_scale, interpolation=cv2.INTER_CUBIC),
            flip_and_color_jitter,
            A.GaussianBlur(p=0.1),
            A.Solarize(p=0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_crop = A.Compose([
            A.RandomResizedCrop(96, 96, scale=local_crops_scale, interpolation=cv2.INTER_CUBIC),
            flip_and_color_jitter,
            A.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = [self.global_crop1(image=image)['image'],
                 self.global_crop2(image=image)['image']]
        for _ in range(self.local_crops_number):
            crops += [self.local_crop(image=image)['image']]
        return crops