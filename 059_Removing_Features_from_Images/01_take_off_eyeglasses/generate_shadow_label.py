import os
import torch
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image

def get_sample_list(data_dir, keyword="-all.png"):
    all_list = os.listdir(data_dir)

    sample_list = []
    for img_name in all_list:
        if img_name.find(keyword) == -1:
            continue
        strs = img_name.split('-')
        sample_name = '-'.join(strs[:-1])
        sample_list.append(sample_name)
        
    return sample_list

def generate_shadow_mask_label(img_with_shadow, img_without_shadow, threshold=0.1):
    diff = torch.abs(img_with_shadow - img_without_shadow)
    diff = diff[0:1, :, :] * 0.3 + diff[1:2, :, :] * 0.59 + diff[2:3, :, :] * 0.11
    all_true = torch.ones(*diff.shape).to(diff.device)
    all_false = torch.zeros(*diff.shape).to(diff.device)
    label = torch.where(diff > threshold, all_true, all_false)

    return label

def save_shadow_mask_label(tensor, save_path):
    out = tensor.cpu().numpy().transpose(1, 2, 0)
    out = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)
    out = out[:, :, 0]
    out_PIL = Image.fromarray(out)
    out_PIL.save(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--syndata_dir", type=str, required=True, help="synthetic dataset dir")
    args = parser.parse_args()
    print("Call with args:")
    print(args)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    sample_list = get_sample_list(args.syndata_dir, keyword="-all.png")
    for sample_name in sample_list:
        # with glass occlusion
        img_1_name = sample_name + '-all.png'
        img_2_name = sample_name + '-glass.png'

        # without glass occlusion
        # img_1_name = sample_name + '-shadow.png'
        # img_2_name = sample_name + '-face.png'

        img_1 = transform(Image.open(os.path.join(args.syndata_dir, img_1_name)))
        img_2 = transform(Image.open(os.path.join(args.syndata_dir, img_2_name)))

        smask_label = generate_shadow_mask_label(img_1, img_2)
        save_name = sample_name + '-shseg.png'
        save_shadow_mask_label(smask_label, os.path.join(args.syndata_dir, save_name))
        