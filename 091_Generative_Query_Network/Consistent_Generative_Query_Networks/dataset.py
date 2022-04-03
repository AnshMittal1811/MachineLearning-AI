import collections, glob, io, os, random

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import ToTensor, Resize, CenterCrop

Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])

def photo_path_from_view(root_path, render_path, view):
    photo_path = os.path.join(render_path,'photo')
    image_path = os.path.join(photo_path,'{0}.jpg'.format(view.frame_num))
    return os.path.join(root_path,image_path)

def transform_viewpoint(v):
    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    # position, [yaw, pitch]
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat


class GQNDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, "{}.pt".format(idx))
        data = torch.load(scene_path)

        byte_to_tensor = lambda x: ToTensor()(Resize(64)((Image.open(io.BytesIO(x)))))


        images = torch.stack([byte_to_tensor(frame) for frame in data.frames])

        viewpoints = torch.from_numpy(data.cameras)
        viewpoints = viewpoints.view(-1, 5)

        if self.transform:
            images = self.transform(images)

        if self.target_transform:
            viewpoints = self.target_transform(viewpoints)

        return images, viewpoints
    
def sample_batch(x_data, v_data, D, M=None, test=False, seed=None):
    random.seed(seed)
    
#     x_q, v_q = x_data, v_data
    if D == "Room":
        K = 10
    elif D == "Jaco":
        K = 7
    elif D == "Labyrinth":
        K = 20
    elif D == "Shepard-Metzler":
        K = 15
    else:
        K = x_data.size(1)

    idx = random.sample(range(x_data.size(1)), K)

    # Sample view
    x, v = x_data[:, idx], v_data[:, idx]
    
    return x, v