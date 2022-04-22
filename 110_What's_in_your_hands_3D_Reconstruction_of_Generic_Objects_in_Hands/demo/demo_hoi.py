"""
Usage:
python -m demo.demo_hoi -e xxx  --image xxx.png [--weight ....ckpt] [--out ]
save prediction to: out/xxx_v0.png, out/xxx_v1.png, 
"""

import functools
from tqdm import tqdm
import argparse
import os.path as osp
from omegaconf.omegaconf import OmegaConf

import torch 
from torch.utils.data.dataloader import DataLoader
from pytorch3d.renderer.cameras import PerspectiveCameras

from config.defaults import get_cfg_defaults
from datasets.custom import Custom
from nnutils.hand_utils import ManopthWrapper, get_nTh
from nnutils import model_utils
from nnutils import geom_utils, mesh_utils, image_utils



def recon_predict(args):
    """save HOI obj"""

    device = 'cuda:0'

    # get config
    cfg_def = get_cfg_defaults()
    cfg_def = OmegaConf.create(cfg_def.dump())
    cfg = OmegaConf.load(osp.join(args.experiment_directory, 'hparams.yaml'))
    arg_cfg = OmegaConf.from_dotlist(['%s=%s' % (a,b) for a,b in zip(args.opts[::2], args.opts[1::2])])
    cfg = OmegaConf.merge(cfg_def, cfg, arg_cfg)
    cfg.MODEL.BATCH_SIZE = 1

    # get dataloader and dataset 
    dataset = Custom(args.data_dir)
    dataloader = DataLoader(dataset, 1)


    # get model
    model = model_utils.load_model(cfg, args.experiment_directory, 'last')
    hand_wrapper = ManopthWrapper().to(device)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        cam_f, cam_p = batch['cam_f'], batch['cam_p']
        cTh = batch['cTh']
        with torch.no_grad():
            hTn = get_nTh(hand_wrapper, batch['hA'].cuda(), cfg.DB.RADIUS, inverse=True)

            hObj, hHand = forward_to_mesh(
                model, 
                batch['image'], 
                batch['cTh'], 
                batch['hA'], 
                batch['cam_f'], batch['cam_p'], 
                geom_utils.matrix_to_se3(hTn),
                hand_wrapper=hand_wrapper,
                obj_mask=batch['obj_mask'],
                )

        hHand.textures = mesh_utils.pad_texture(hHand, 'blue')
        hHoi = mesh_utils.join_scene([hObj, hHand]).to(device)
        cHoi = mesh_utils.apply_transform(hHoi, cTh.to(device))
        cameras = PerspectiveCameras(cam_f, cam_p, device=device)
        iHoi = mesh_utils.render_mesh(cHoi, cameras,)
        image_utils.save_images(iHoi['image'], osp.join(args.out, 'cHoi', batch['index'][0]),
            bg=batch['image'], mask=iHoi['mask'])
        
        image_list = mesh_utils.render_geom_rot(cHoi, cameras=cameras, view_centric=True)
        image_utils.save_gif(image_list, osp.join(args.out, 'hHoi', batch['index'][0]))

        mesh_utils.dump_meshes([osp.join(args.out, 'meshes', batch['index'][0])], hHoi)



def forward_to_mesh(model, images, cTh,hA, cam_f, cam_p,hTn, hand_wrapper, obj_mask=None):
    batch = {
        'cam_f': cam_f,
        'cam_p': cam_p,
        'image': images,
        'cTh': cTh,
        'obj_mask': obj_mask,
        'hA': hA,
        'hTn': hTn
    }

    batch = model_utils.to_cuda(batch)
    device = batch['image'].device
    hHand, hJoints = hand_wrapper(None, batch['hA'], mode='inner')

    image_feat = model.enc(batch['image'], mask=batch['obj_mask'])  # (N, D, H, W)

    hTx = batch['hTn']
    cTx = geom_utils.compose_se3(batch['cTh'], hTx)

    hTjs = hand_wrapper.pose_to_transform(batch['hA'], False)  # (N, J, 4, 4)
    N, num_j, _, _ = hTjs.size()
    jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
    hTx = geom_utils.se3_to_matrix(hTx
            ).unsqueeze(1).repeat(1, num_j, 1, 1)
    jsTx = jsTh @ hTx

    out = {'z': image_feat, 'jsTx': jsTx}

    camera = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
    cTx = geom_utils.compose_se3(batch['cTh'], batch['hTn'])
    # normal space, joint space jsTn, image space 
    sdf = functools.partial(model.dec, z=out['z'], hA=batch['hA'], 
        jsTx=out['jsTx'].detach(), cTx=cTx.detach(), cam=camera)
    # TODO: handel empty predicdtion
    xObj = mesh_utils.batch_sdf_to_meshes(sdf, N, bound=True)

    hTx = batch['hTn']
    hObj = mesh_utils.apply_transform(xObj, hTx)
    return hObj, hHand

  

def parse_args():
    arg_parser = argparse.ArgumentParser(description="Demo args")
    arg_parser.add_argument('--data_dir', default='/checkpoint/yufeiy2/hoi_output/test_data', type=str)
    arg_parser.add_argument('--out', default='/checkpoint/yufeiy2/ihoi_out', type=str)
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        default='/checkpoint/yufeiy2/hoi_output/release_model/mow'
    )
    arg_parser.add_argument("opts",  default=None, nargs=argparse.REMAINDER)
    return arg_parser


if __name__ == '__main__':
    arg_parser = parse_args()
    args = arg_parser.parse_args()
    recon_predict(args)
