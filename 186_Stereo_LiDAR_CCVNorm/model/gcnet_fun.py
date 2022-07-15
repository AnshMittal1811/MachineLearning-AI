"""
Ref: https://github.com/wyf2017/DSMnet/blob/master/models/util_fun.py
"""
import torch

# cat with slicing same h*w
def myCat2d(*seq):
    assert len(seq[0].shape) == 4
    bn, c, h, w = seq[0].shape
    for tmp in seq:
        _, _, ht, wt = tmp.shape
        if(h > ht): h = ht
        if(w > wt): w = wt
    seq1 = [ seq[i][:, :, :h, :w] for i in range(len(seq))]
    return torch.cat(seq1, dim = 1)


# cat with slicing same d*h*w
def myCat3d(*seq):
    assert len(seq[0].shape) == 5
    bn, c, d, h, w = seq[0].shape
    for tmp in seq:
        _, _, dt, ht, wt = tmp.shape
        if(d > dt): d = dt
        if(h > ht): h = ht
        if(w > wt): w = wt
    seq1 = [ seq[i][:, :, :d, :h, :w] for i in range(len(seq))]
    return torch.cat(seq1, dim = 1)


# add with slicing same h*w
def myAdd2d(tensor1, tensor2):
    assert len(tensor1.shape) == 4
    _, _, h1, w1 = tensor1.shape
    _, _, h2, w2 = tensor2.shape
    h = min(h1, h2)
    w = min(w1, w2)
    tensor1 = tensor1[:, :, :h, :w]
    tensor2 = tensor2[:, :, :h, :w]
    return tensor1 + tensor2


# add with slicing same d*h*w
def myAdd3d(tensor1, tensor2):
    assert len(tensor1.shape) == 5
    _, _, d1, h1, w1 = tensor1.shape
    _, _, d2, h2, w2 = tensor2.shape
    d = min(d1, d2)
    h = min(h1, h2)
    w = min(w1, w2)
    tensor1 = tensor1[:, :, :d, :h, :w]
    tensor2 = tensor2[:, :, :d, :h, :w]
    return tensor1 + tensor2


# get max with slicing same h*w
def myMax2d(disp_low, disp_high):
    assert len(disp_low.shape) == 4
    bn, c, h, w = disp_high.shape
    mask = disp_high < disp_low[:bn, :c, :h, :w]
    disp_high[mask] = disp_low[:bn, :c, :h, :w][mask]
    return disp_high


# simple test
def test():
    tensor1 = torch.rand(1, 1, 65, 65)
    tensor2 = torch.rand(1, 1, 66, 66)
    tensor3 = torch.rand(1, 6, 64, 64)
    y1 = myCat2d(tensor1, tensor2, tensor3)
    y2 = myAdd2d(tensor1, tensor2)
    print(y1.shape, y2.shape)
    
    tensor1 = torch.rand(1, 1, 9, 65, 65)
    tensor2 = torch.rand(1, 1, 10, 66, 66)
    tensor3 = torch.rand(1, 6, 8, 64, 64)
    y1 = myCat3d(tensor1, tensor2, tensor3)
    y2 = myAdd3d(tensor1, tensor2)
    print(y1.shape, y2.shape)
