import torch.nn as nn

class DummyModel(nn.Module):

    def __init__(self, dummy_param="foo"):
        super(DummyModel, self).__init__()
        self.dummy_param=dummy_param
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':
    dummy = DummyModel()
    print(dummy)

    for name, param in dummy.named_parameters():
        if param.requires_grad:
            print("param: {} requires_grad: {}".format(name, param.requires_grad))