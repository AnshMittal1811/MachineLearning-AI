from torch.utils import data

class ATISData(data.Dataset):
    def __init__(self, X, y):
        self.len = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
             
    def __len__(self):
        return self.len

