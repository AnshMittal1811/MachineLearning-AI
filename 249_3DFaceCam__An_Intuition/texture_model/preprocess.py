from torch.utils.data import Dataset
import random
import PIL
import torch
from torchvision import transforms
import numpy as np

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def imagefolder_loader(path):
    def loader(transform):
        data = datasets.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size, num_workers=4)
        return data_loader
        
    return loader


def sample_data(dataloader, image_size=4):
    transform = transforms.Compose([transforms.Resize(image_size),
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    loader = dataloader(transform)

    return loader


class my_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value

class MyDataset(Dataset):
    def __init__(self, zid_dict, image_paths, transform=None, img_size=4):
        random.shuffle(image_paths)
        self.image_paths = image_paths
        self.transform = transforms.Resize(img_size)
        self.img_size = img_size
        self.zid_dict = zid_dict
        
    
    def get_target_from_path(self, path):
        # Implement your target extraction here
        
        y_id = path.split('/')[-2]
        y_exp = (path.split('/')[-1]).split('_')[0]
        
        return int(y_id), int(y_exp)
    
    def __getitem__(self, index):

        x = PIL.Image.open(self.image_paths[index])

        x = torch.from_numpy(np.asarray(x))
        x = x.to(torch.float)

        x = x.permute(2,0,1)
        
        y_id, y_exp = self.get_target_from_path(self.image_paths[index])
        
        z_id = self.zid_dict[y_id]
        
        if self.transform:
            x = self.transform(x)
        
        return x/255.0, y_id-1, y_exp, z_id
    
    def __len__(self):
        return len(self.image_paths)

