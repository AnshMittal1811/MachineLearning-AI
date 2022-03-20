import torch
import torch.nn.functional as F
import torch.optim as optim

def random_color_palette(k):
    # Create a palette of K maximally distant colors
    pts = (torch.rand(k, 3)*2 - 1).requires_grad_()
    optimizer = optim.SGD([pts], lr=0.1)
    for i in range(10000):
        optimizer.zero_grad()
        loss = -torch.pdist(F.normalize(pts, dim=-1)).mean()
        loss.backward()
        optimizer.step()
        pts.data[0] = torch.Tensor([0, 0, 1])
        pts.data.clamp_(min=-1, max=1)
    return pts.data