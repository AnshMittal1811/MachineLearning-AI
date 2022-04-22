import functools
import os.path as osp
from omegaconf.omegaconf import OmegaConf
from pytorch3d.renderer import PerspectiveCameras
import torch

from config.defaults import get_cfg_defaults
from nnutils import model_utils

from nnutils import mesh_utils, image_utils, geom_utils
from nnutils.hand_utils import ManopthWrapper, get_nTh



def get_hoi_predictor(args):
    cfg_def = get_cfg_defaults()
    cfg_def = OmegaConf.create(cfg_def.dump())
    cfg = OmegaConf.load(osp.join(args.experiment_directory, 'hparams.yaml'))
    arg_cfg = OmegaConf.from_dotlist(['%s=%s' % (a,b) for a,b in zip(args.opts[::2], args.opts[1::2])])
    cfg = OmegaConf.merge(cfg_def, cfg, arg_cfg)
    cfg.MODEL.BATCH_SIZE = 1
    model = model_utils.load_model(cfg, args.experiment_directory, 'last')

    predictor = Predictor(model)
    return predictor


class Predictor:
    def __init__(self,model,):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.model = model.to(device)
        self.hand_wrapper = ManopthWrapper().to(device)
    
    def forward_to_mesh(self, batch):
        model = self.model
        cfg = self.model.cfg
        hand_wrapper = self.hand_wrapper

        batch = model_utils.to_cuda(batch, self.device)

        hTx = geom_utils.matrix_to_se3(
            get_nTh(hand_wrapper, batch['hA'].cuda(), cfg.DB.RADIUS, inverse=True))

        device = self.device
        hHand, hJoints = hand_wrapper(None, batch['hA'], mode='inner')

        image_feat = model.enc(batch['image'], mask=batch['obj_mask'])  # (N, D, H, W)

        cTx = geom_utils.compose_se3(batch['cTh'], hTx)

        hTjs = hand_wrapper.pose_to_transform(batch['hA'], False)  # (N, J, 4, 4)
        N, num_j, _, _ = hTjs.size()
        jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
        hTx_exp = geom_utils.se3_to_matrix(hTx
                ).unsqueeze(1).repeat(1, num_j, 1, 1)
        jsTx = jsTh @ hTx_exp

        out = {'z': image_feat, 'jsTx': jsTx}

        camera = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
        cTx = geom_utils.compose_se3(batch['cTh'], hTx)
        # normal space, joint space jsTn, image space 
        sdf = functools.partial(model.dec, z=out['z'], hA=batch['hA'], 
            jsTx=out['jsTx'].detach(), cTx=cTx.detach(), cam=camera)
        # TODO: handel empty predicdtion
        xObj = mesh_utils.batch_sdf_to_meshes(sdf, N, bound=True)

        hObj = mesh_utils.apply_transform(xObj, hTx)
        out['hObj'] = hObj
        out['hHand'] = hHand
        return out


def vis_hand_object(output, data, image, save_dir):
    hHand = output['hHand']
    hObj = output['hObj']
    device = hObj.device

    cam_f, cam_p = data['cam_f'], data['cam_p']
    cTh = data['cTh']

    hHand.textures = mesh_utils.pad_texture(hHand, 'blue')
    hHoi = mesh_utils.join_scene([hObj, hHand]).to(device)
    cHoi = mesh_utils.apply_transform(hHoi, cTh.to(device))
    cameras = PerspectiveCameras(cam_f, cam_p, device=device)
    iHoi = mesh_utils.render_mesh(cHoi, cameras,)
    image_utils.save_images(iHoi['image'], save_dir + '_cHoi', bg=data['image']/2+0.5, mask=iHoi['mask'])
    image_utils.save_images(data['image']/2+0.5, save_dir + '_inp')

    image_list = mesh_utils.render_geom_rot(cHoi, cameras=cameras, view_centric=True)
    image_utils.save_gif(image_list, save_dir + '_cHoi')

    mesh_utils.dump_meshes([save_dir + '_hoi'], hHoi)


