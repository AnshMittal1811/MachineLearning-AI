import torch
from torchvision import datasets, transforms
import  numpy as np

class CostumDataset(object):
    def __init__(self, args):
        kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

        Data=np.load(args.dataset)
        Feature=Data[:,0:78951]
        Label_id = Data[:,78951]
        Label_ex = Data[:, 78952]
        tensor_x = torch.Tensor(Feature)  # transform to torch tensor
        tensor_y = torch.Tensor(Label_id)
        tensor_z = torch.Tensor(Label_ex)

        #Original
        trainset = torch.utils.data.TensorDataset(tensor_x, tensor_y,tensor_z)  # create your datset

        self.train_loader = torch.utils.data.DataLoader(trainset,  batch_size=args.batch_size, shuffle=True,**kwargs)
        self.test_loader = None
