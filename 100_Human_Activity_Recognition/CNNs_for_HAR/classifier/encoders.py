
import torch
from torch import nn

def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("{}".format(model_type))

class Cnn1(nn.Module):
    def __init__(self, data_size, n_classes):
        """
        """
        super(Cnn1, self).__init__()
        self.n_chan = data_size[0]
        self.n_classes = n_classes

        # Convolutional Layers
        self.conv1 = nn.Conv1d(self.n_chan, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        self.drop = nn.Dropout(p=0.6)
        self.pool = nn.MaxPool1d(kernel_size=2,stride=2)

        # Fully connected layers
        self.lin3 = nn.Linear(3968, 100)
        self.lin4 = nn.Linear(100, self.n_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        a = torch.relu(self.conv1(x))
        a = torch.relu(self.conv2(a))
        a = self.drop(a)
        a = self.pool(a)
        #Fully connected layers
        a = a.view((batch_size, -1))
        a = torch.relu(self.lin3(a))
        a = torch.relu(self.lin4(a))

        return a

class Cnn2(nn.Module):
    def __init__(self, data_size, n_classes):
        """
        """
        super(Cnn2, self).__init__()
        self.n_chan = data_size[0]
        self.n_classes = n_classes

        # Convolutional Layers
        self.conv1 = nn.Conv1d(self.n_chan, 64, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, stride=1)
        self.drop = nn.Dropout(p=0.6)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.lin3 = nn.Linear(3840, 100)
        self.lin4 = nn.Linear(100, self.n_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        a = torch.relu(self.conv1(x))
        a = torch.relu(self.conv2(a))
        a = self.drop(a)
        a = self.pool(a)
        #Fully connected layers
        a = a.view((batch_size, -1))
        a = torch.relu(self.lin3(a))
        a = torch.relu(self.lin4(a))

        return a

