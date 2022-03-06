import torch.nn as nn
import torch.nn.functional as F
import torch

class DQN_NN_Naive(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()

        self.fc1 = nn.Linear(in_features=img_height * img_width * 3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


class DQN_CNN_2013(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super().__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 16, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(16, 32, kernel_size=4, stride=2),
                                        nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(9*9*32, 256),
                                        nn.ReLU(True),
                                        nn.Linear(256, num_classes)
                                        )
        # nn.Dropout(0.3),  # BZX: optional [TRY]
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

class DQN_CNN_2015(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super().__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

#TODO
class Dueling_DQN_2016_Modified(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super().__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4,bias=False),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2,bias=False),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1,bias=False),
                                        nn.ReLU(True),
                                        nn.Conv2d(64,1024,kernel_size=7,stride=1,bias=False),
                                        nn.ReLU(True)
                                        )
        self.streamA = nn.Linear(512, num_classes)
        self.streamV = nn.Linear(512, 1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.cnn(x)
        sA,sV = torch.split(x,512,dim = 1)
        sA = torch.flatten(sA,start_dim=1)
        sV = torch.flatten(sV, start_dim=1)
        sA = self.streamA(sA) #(B,4)
        sV = self.streamV(sV) #(B,1)
        # combine this 2 values together
        Q_value = sV + (sA - torch.mean(sA,dim=1,keepdim=True))
        return Q_value #(B,4)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
