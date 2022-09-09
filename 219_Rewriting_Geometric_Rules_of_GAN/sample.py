import os
import sys
import argparse
import cv2
import imageio
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from util import tile_images


def generate(args, g_ema, device):
    with torch.no_grad():
        g_ema.eval()
        if args.fixed_z is not None:
            sample_z = torch.load(args.fixed_z, map_location=device)
            sample = g_ema(sample_z, None, truncation_psi=args.truncation)
            samples = sample.cpu().numpy()

        else:
            z = torch.from_numpy(np.random.RandomState(args.seed).randn(args.samples, g_ema.z_dim)).to(device)
            samples = []
            for start in tqdm(range(0, args.samples, args.batch_size)):
                end = min(args.samples, start + args.batch_size)
                sample_z = z[start:end]
                sample = g_ema(sample_z, None, truncation_psi=args.truncation)
                samples.append(sample.cpu().numpy())
            samples = np.concatenate(samples)
    
    samples = np.transpose((samples + 1) / 2, (0, 2, 3, 1))
    samples = np.clip(samples * 255 + 0.5, 0, 255).astype(np.uint8)
    return samples


pretrained_models = {
    'cat': './pretrained/stylegan3-r-afhqv2cat-512x512.pkl',
    'dog': './pretrained/stylegan3-r-afhqv2dog-512x512.pkl',
    'wild': './pretrained/stylegan3-r-afhqv2wild-512x512.pkl',
    'horse': './pretrained/stylegan3-r-horse-256x256.pkl',
    'house': './pretrained/stylegan3-r-house-512x512.pkl',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', type=str, help='output path. Folder if tile is false, and a file if tile is true.')
    parser.add_argument('-m', '--ckpt', type=str, help='checkpoint file to the edited model.')
    parser.add_argument('--fixed_z', type=str, default=None, help='use a pre-stored .pth file as latent z.')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--samples', type=int, default=32, help='how many samples to generate, if fixed_z is None.')
    parser.add_argument('--truncation', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=6000)
    parser.add_argument('--pretrained', default='cat')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('-t', '--tile', action='store_true', help='tile the samples, the following options only works when tile is true.')
    parser.add_argument('--row_im', type=int, default=8, help='how many images in a row.')
    parser.add_argument('--video', action='store_true', help='option to render a video.')
    parser.add_argument('--max_height', type=int, default=1024, help='max height of the tiled images.')
    parser.add_argument('--video_width', type=int, default=1280, help='width of a video.')
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--fps', type=int, default=80)

    args = parser.parse_args()
    device = args.device
    torch.set_grad_enabled(False)

    # add stylegan3 to the syspath and import the libraries
    sys.path.append('./models/networks/stylegan3')
    import dnnlib
    import legacy

    # load the model
    pretrained_pkl = pretrained_models[args.pretrained]
    with dnnlib.util.open_url(pretrained_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    G.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)

    # generate samples
    samples = generate(args, G, device)

    # if not tiling, save individual images to a folder, and then exit.
    if not args.tile:
        os.makedirs(args.output, exist_ok=True)
        for ind, img in enumerate(samples):
            Image.fromarray(img).save(f"{args.output}/{ind}.png")
        exit()

    # Tile the images, and downsample it to args.max_height if too large
    image = tile_images(samples, picturesPerRow=args.row_im)
    h, w = image.shape[:2]
    nh, nw = args.max_height, int(w * args.max_height / h)
    image = cv2.resize(image, (nw, nh))

    # Render the output as an image or a video
    if not args.video:
        Image.fromarray(image).save(args.output)
    else:
        tot_width = image.shape[1]
        image = np.hstack([image, image[:, :args.video_width]]) # repeat the first frame to make video loopable
        ims = []
        for start in range(0, tot_width+1, args.stride):
            end = start + args.video_width
            ims.append(image[:, start:end])

        video = imageio.get_writer(args.output, mode='I', fps=args.fps, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{args.output}')
        for im in ims:
            video.append_data(im)
        video.close()
