import os
import sys
import argparse
import imageio
import torch
import numpy as np
from PIL import Image


def load_model(sg3_repo, base_model_path, edited_model_path, device):
    # import stylegan repo
    sys.path.append(sg3_repo)
    import dnnlib
    import legacy

    # load the base model
    with dnnlib.util.open_url(base_model_path) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # overwrite the state_dict with new weights
    state_dict = G.state_dict()
    new_params = torch.load(edited_model_path, map_location=device)
    for name, param in new_params.items():
        state_dict[name].copy_(param)

    return G


def ganspace_edit(w, comp_idx, s_range, stylemix, device='cuda'):
    start, end = stylemix

    shift = torch.from_numpy(pcs_scaled[comp_idx])[None, None, :].to(device)
    strength = torch.from_numpy(np.array(s_range))[:, None, None].to(device)
    latent = w.clone().repeat(*strength.shape)
    latent[:, start:end, :] = latent[:, start:end, :] + strength * shift
    ims = G.synthesis(latent)    
    ims = torch.clip((ims + 1) * 127.5, 0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
 
    return ims


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ganspace', required=True, help='path to the precomputed ganspace directions')
    parser.add_argument('--comp_idx', type=int, required=True, help="which principle component to use")
    parser.add_argument('--strength', type=float, required=True, help="strength applied to the latent shift, value can be negative")
    parser.add_argument('--layers', type=str, required=True, help="layers to apply GANSpace (e.g., 3,6 means layer 3 to 5")
    parser.add_argument('--num_frames', type=int, default=10, help="number of frames to render the edit transition")

    parser.add_argument('--save_dir', type=str, default='./output', help="place to save the output")
    parser.add_argument('--save_video', action='store_true', help='option to render to edit transtions into a video as well')
    parser.add_argument('--video_fps', type=int, default=20, help='fps of the rendered video')
    parser.add_argument('--base_model_path', type=str, required=True, help=".pkl file to the base stylegan generator weights")
    parser.add_argument('--edited_model_path', type=str, required=True, help=".pth file to the edited model path")
    parser.add_argument('--fixed_z', type=str, default=None, help="expect a .pth file. If given, will use this file as the input noise for the output")
    parser.add_argument('--fixed_w', type=str, default=None, help="expect a .pth file. If given, will use this file as the latent w for the output")
    parser.add_argument('--trunc', type=float, default=1.0, help='amount of truncation applied to the samples')
    parser.add_argument('--samples', type=int, default=5, help="number of samples to generate, will be overridden if --fixed_z or --fixed_w is given")
    parser.add_argument('--seed', type=int, default=8000, help="if specified, use a fixed random seed")

    parser.add_argument('--stylegan_path', default='./models/networks/stylegan3', help='path to stylegan repo')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    device = args.device
    torch.set_grad_enabled(False)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load networks.
    G = load_model(args.stylegan_path, args.base_model_path, args.edited_model_path, device)

    # get the latent samples before the edits
    if args.fixed_z is None and args.fixed_w is None:
        z = np.random.RandomState(args.seed).randn(args.samples, G.z_dim)
        z = torch.from_numpy(z).to(device)
        latents = G.mapping(z, None, truncation_psi=args.trunc)
    elif args.fixed_z is None:
        latents = torch.load(args.fixed_w, map_location=device)
    elif args.fixed_w is None:
        z = torch.load(args.fixed_z, map_location=device)
        latents = G.mapping(z, None, truncation_psi=args.trunc)
    else:
        raise KeyError('--fixed_w and --fixed_z tags cannot be both specified')

    # setup GANSpace config
    ganspace = np.load(args.ganspace)
    pcs_scaled = ganspace['pcs_scaled']

    comp_idx, strength = args.comp_idx, args.strength
    stylemix = [int(d) for d in args.layers.split(',')]
    s_range = np.linspace(0, strength, args.num_frames)

    for idx in range(latents.size(0)):
        ims = ganspace_edit(latents[idx], comp_idx, s_range, stylemix, device)

        # render results
        outdir = os.path.join(args.save_dir, str(idx))
        os.makedirs(outdir, exist_ok=True)

        im_stack = np.hstack(ims[::4])
        Image.fromarray(im_stack).save(f'{outdir}/stack.png')

        for k, im in enumerate(ims):
            Image.fromarray(im).save(f'{outdir}/{k}.png')

        if args.save_video:
            video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=args.video_fps, codec='libx264', bitrate='16M')
            print (f'Saving optimization progress video "{outdir}/proj.mp4"')
            for im in ims:
                video.append_data(im)
            video.close()
