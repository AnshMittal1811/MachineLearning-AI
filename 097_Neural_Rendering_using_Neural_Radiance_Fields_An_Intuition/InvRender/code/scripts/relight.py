import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
from PIL import Image
import math

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from model.sg_render import compute_envmap
import imageio


tonemap_img = lambda x: np.power(x, 1./2.2)
clip_img = lambda x: np.clip(x, 0., 1.)


def decode_img(img, batch_size, total_pixels, img_res, is_tonemap=False):
    img = img.reshape(batch_size, total_pixels, 3)
    img = plt.lin2img(img, img_res).detach().cpu().numpy()[0]
    img = img.transpose(1, 2, 0)
    if is_tonemap:
        img = tonemap_img(img)
    img = clip_img(img)
    return img


def relit_with_light(model, relit_dataloader, images_dir, 
                total_pixels, img_res, albedo_ratio=None, light_type='origin'):
    
    all_frames = []

    for data_index, (indices, model_input, ground_truth) in enumerate(relit_dataloader):
        print('relighting data_index: ', data_index, len(relit_dataloader))
        for key in model_input.keys():
            model_input[key] = model_input[key].cuda()

        split = utils.split_input(model_input, total_pixels)
        res = []
        for s in split:
            s['albedo_ratio'] = albedo_ratio
            out = model(s, trainstage="Material")
            res.append({
                'normals': out['normals'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'object_mask': out['object_mask'].detach(),
                'roughness':out['roughness'].detach(),
                'diffuse_albedo': out['diffuse_albedo'].detach(),
                'sg_diffuse_rgb': out['sg_diffuse_rgb'].detach(),
                'sg_specular_rgb': out['sg_specular_rgb'].detach(),
                'indir_rgb':out['indir_rgb'].detach(),
                'sg_rgb': out['sg_rgb'].detach(),
                'bg_rgb': out['bg_rgb'].detach(),
            })
        
        out_img_name = '{}'.format(indices[0])
        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, total_pixels, batch_size)

        assert (batch_size == 1)

        # input mask
        mask = model_input['object_mask']
        mask = plt.lin2img(mask.unsqueeze(-1), img_res)[0].permute(1, 2, 0)

        # render background
        bg_rgb = model_outputs['bg_rgb']
        bg_rgb = decode_img(bg_rgb, batch_size, total_pixels, img_res, is_tonemap=True)
        object_mask = model_outputs['network_object_mask'].unsqueeze(0)
        object_mask = plt.lin2img(object_mask.unsqueeze(-1), img_res)[0].permute(1, 2, 0)
    
        ### save sg
        if light_type == 'origin':
            rgb_relit = model_outputs['sg_rgb'] + model_outputs['indir_rgb']
        else:
            rgb_relit = model_outputs['sg_rgb'] 
        rgb_relit = decode_img(rgb_relit, batch_size, total_pixels, img_res, is_tonemap=True)

        # envmap background
        bg_mask = ~object_mask.expand(-1,-1,3).cpu().numpy()
        rgb_relit[bg_mask] = bg_rgb[bg_mask]
        rgb_relit_env_bg = Image.fromarray((rgb_relit * 255).astype(np.uint8))
        rgb_relit_env_bg.save('{0}/sg_rgb_bg_{1}.png'.format(images_dir, out_img_name))
        all_frames.append(np.array(rgb_relit))

        if light_type == 'origin':
            ### save roughness
            roughness_relit = model_outputs['roughness']
            roughness_relit = decode_img(roughness_relit, batch_size, total_pixels, img_res, is_tonemap=False)
            roughness_relit = Image.fromarray((roughness_relit * 255).astype(np.uint8))
            roughness_relit.save('{0}/roughness_{1}.png'.format(images_dir, out_img_name))

            ### save diffuse albedo
            albedo_relit = model_outputs['diffuse_albedo']
            albedo_relit = decode_img(albedo_relit, batch_size, total_pixels, img_res, is_tonemap=True)
            albedo_relit = Image.fromarray((albedo_relit * 255).astype(np.uint8))
            albedo_relit.save('{0}/albedo_{1}.png'.format(images_dir, out_img_name))

            ### save normals
            normal = model_outputs['normals']
            normal = (normal + 1.) / 2.
            normal = decode_img(normal, batch_size, total_pixels, img_res, is_tonemap=False)
            normal = Image.fromarray((normal * 255).astype(np.uint8))
            normal.save('{0}/normal_{1}.png'.format(images_dir, out_img_name))

            ### save indirect rendering
            # indir_rgb = model_outputs['indir_rgb']
            # indir_rgb = decode_img(indir_rgb, batch_size, total_pixels, img_res, is_tonemap=True)
            # indir_rgb = Image.fromarray((indir_rgb * 255).astype(np.uint8))
            # indir_rgb.save('{0}/sg_indir_rgb_{1}.png'.format(images_dir, out_img_name))

    imageio.mimwrite(os.path.join(images_dir, 'video_rgb.mp4'), all_frames, fps=20, quality=9)
    print('Done rendering', images_dir)


def relight_obj(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    relits_folder_name = kwargs['relits_folder_name']

    expname = 'Mat-' + kwargs['expname']

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', relits_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    relitdir = os.path.join('../', relits_folder_name, expname, os.path.basename(kwargs['data_split_dir']))

    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()
    
    # load data
    relit_dataset = utils.get_class(conf.get_string('train.dataset_class'))(
                                    kwargs['data_split_dir'], kwargs['frame_skip'], split='test')
    relit_dataloader = torch.utils.data.DataLoader(relit_dataset, batch_size=1,
                                    shuffle=False, collate_fn=relit_dataset.collate_fn)

    total_pixels = relit_dataset.total_pixels
    img_res = relit_dataset.img_res

    # load trained model
    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
    print('Loading checkpoint: ', ckpt_path)
    saved_model_state = torch.load(ckpt_path)
    model.load_state_dict(saved_model_state["model_state_dict"])

    print("start render...")
    model.eval()

    # images_dir = relitdir
    # utils.mkdir_ifnotexists(images_dir)
    # print('Output directory is: ', images_dir)

    # with open(os.path.join(relitdir, 'ckpt_path.txt'), 'w') as fp:
    #     fp.write(ckpt_path + '\n')

    # relit_with_light(model, relit_dataloader, images_dir, 
    #                 total_pixels, img_res, albedo_ratio=None, light_type='origin')


    envmap6_path = './envmaps/envmap6'
    print('Loading light from: ', envmap6_path)
    model.envmap_material_network.load_light(envmap6_path)
    images_dir = relitdir + '_envmap6_relit'
    utils.mkdir_ifnotexists(images_dir)
    print('Output directory is: ', images_dir)

    relit_with_light(model, relit_dataloader, images_dir, 
                    total_pixels, img_res, albedo_ratio=None, light_type='envmap6')
    
    envmap12_path = './envmaps/envmap12'
    print('Loading light from: ', envmap12_path)
    model.envmap_material_network.load_light(envmap12_path)
    images_dir = relitdir + '_envmap12_relit'
    utils.mkdir_ifnotexists(images_dir)
    print('Output directory is: ', images_dir)
    relit_with_light(model, relit_dataloader, images_dir, 
                    total_pixels, img_res, albedo_ratio=None, light_type='envmap12')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/default.conf')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be relituated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')

    parser.add_argument('--data_split_dir', type=str, default='')
    parser.add_argument('--frame_skip', type=int, default=1, help='skip frame when test')

    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')

    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')

    opt = parser.parse_args()

    gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    relight_obj(conf=opt.conf,
                relits_folder_name='relits',
                data_split_dir=opt.data_split_dir,
                expname=opt.expname,
                exps_folder_name=opt.exps_folder,
                timestamp=opt.timestamp,
                checkpoint=opt.checkpoint,
                frame_skip=opt.frame_skip
                )
