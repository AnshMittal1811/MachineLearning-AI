from torch.utils.data import DataLoader, Dataset
import torch


class EViTPair(Dataset):

    def __init__(self, img_data, huffman_feature, transform=None):
        self.img_data = img_data
        self.transform = transform
        self.huffman_feature = huffman_feature

    def __getitem__(self, index):
        img = self.img_data[index]
        huffman = torch.tensor(self.huffman_feature[index])

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2, huffman

    def __len__(self):
        return len(self.huffman_feature)