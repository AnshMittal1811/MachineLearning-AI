import torch
from timesformer_pytorch import TimeSformer

model = TimeSformer(
    dim = 512,
    image_size = 224,
    patch_size = 16,
    num_frames = 8,
    num_classes = 10,
    depth = 12,
    heads = 8,
    dim_head =  64,
    attn_dropout = 0.1,
    ff_dropout = 0.1
)

video = torch.randn(2, 8, 3, 224, 224) # (batch x frames x channels x height x width)
mask = torch.ones(2, 8).bool() # (batch x frame) - use a mask if there are variable length videos in the same batch

pred = model(video, mask = mask) # (2, 10)
