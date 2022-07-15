import torch
import torchvision.transforms as transforms

__imagenet_stats = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}


def color_normalize(normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]

    return transforms.Compose(t_list)


def color_preproccess(normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def get_transform(name='imagenet',
                  normalize=None,
                  augment=True):
    normalize = __imagenet_stats
    if augment:
        return color_preproccess(normalize=normalize)
    else:
        return color_normalize(normalize=normalize)
