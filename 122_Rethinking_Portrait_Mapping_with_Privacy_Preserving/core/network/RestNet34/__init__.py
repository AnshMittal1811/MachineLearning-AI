from .P3mNet import P3mNet


__all__ = ['p3mnet_resnet34']

def p3mnet_resnet34(pretrained=True, **kwargs):
    return P3mNet(pretrained=pretrained)