from builtins import breakpoint
import os
import lpips
import imageio
import argparse 
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from options.options import get_dataset, get_model
from models.networks.sync_batchnorm import convert_model
from models.base_model import BaseModel

def tensor_to_image(image):
    img = torch.clamp(image.permute(1,2,0),min=-1, max=1).detach().cpu().numpy() * 0.5 + 0.5
    img = np.ceil(img * 255) / 255.0
    return img
torch.backends.cudnn.enabled = True

def main(args):

    MODEL_PATH = args.model_path

    opts = torch.load(MODEL_PATH)['opts']
    opts.render_ids = [1]
    opts.input_view_num = args.input_view
    opts.test_view = args.src_list
    name = args.name
    outpath = args.output_path


    model = get_model(opts)

    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
    device = 'cuda:' + str(torch_devices[0])

    if 'sync' in opts.norm_G:
        model = convert_model(model)
        model = nn.DataParallel(model, torch_devices[0:1]).cuda()
    else:
        model = nn.DataParallel(model, torch_devices[0:1]).cuda()


    lpips_vgg = lpips.LPIPS(net="vgg").to(device=device)
    #  Load the original model to be tested
    model_to_test = BaseModel(model, opts)
    model_to_test.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
    model_to_test.eval()

    print("Loaded model")
    dataset = get_dataset(opts)
    test_set = dataset(stage='test', opts=opts)
    test_set.use_depth = True
    dataloader = DataLoader(
        test_set,
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    )
    output_path = os.path.join(args.output_path, args.name)
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)
    # We follow the paradigm of pixel-nerf for evaluation. 
    # The input/src view is pre-defined and there are some pre-defined excluded views.
    # We report performance with and wo these excluded views.
    src_list = args.src_list
    src_list = list(map(int, src_list.split()))
    base_exclude_views = deepcopy(src_list)
    base_exclude_views.extend([3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39])
    evaluation = OrderedDict()
    
    for batch in dataloader:
        scan_name = batch['path'][0].split("/")[-1]
        with torch.no_grad():
            results = model_to_test.model.module.eval_batch(batch, num_view=args.input_view)
        # If we use depth completion, then there is "ComDepth" in results. We save the provided incomplete depths, completed depths, and mse between them.
        # If we directly estimate depth from images, then there is no "ComDepth" and depth ground truth.
        com_depth = "ComDepth" in results[1]
        ## load the input 3 views 
        depth_imgs = []
        if "depths" in batch.keys():

            for i in range(opts.input_view_num):

                if torch.cuda.is_available():
                    depth_imgs.append(batch["depths"][i].cuda())
                else:
                    depth_imgs.append(batch["depths"][i])
        Pred_depth = results[1]["PredDepth"][0:opts.input_view_num]
        H, W = results[1]['PredDepth'][0].shape[-2],results[1]['PredDepth'][0].shape[-1]
        if opts.down_sample:
            depth_imgs = torch.cat(depth_imgs, 1)
            gt_depth = F.interpolate(depth_imgs, size=(H, W), mode="nearest").unsqueeze(2)

        else:
            gt_depth = torch.stack(depth_imgs, 1) # B x num_outputs x 1 x H x W
        gt_depth = gt_depth.contiguous().view(-1, 1, H, W)
        # breakpoint()
        Depth_error = nn.MSELoss()(Pred_depth[gt_depth>0.0001], gt_depth[gt_depth>0.0001])
        input_view1 = batch['images'][0][0] * 0.5 + 0.5
        input_view2 = batch['images'][1][0] * 0.5 + 0.5
        input_view3 = batch['images'][2][0] * 0.5 + 0.5
        input_depth1 = results[1]['PredDepth'][0]
        input_depth2 = results[1]['PredDepth'][1]
        input_depth3 = results[1]['PredDepth'][2]

        input_view1 = input_view1.permute(1,2,0).detach().cpu().numpy()
        input_view2 = input_view2.permute(1,2,0).detach().cpu().numpy()
        input_view3 = input_view3.permute(1,2,0).detach().cpu().numpy()
        input_depth1 = np.uint8(input_depth1[0].detach().cpu().numpy()/input_depth1.max().item() * 255.0)
        input_depth2 = np.uint8(input_depth2[0].detach().cpu().numpy()/input_depth2.max().item() * 255.0)
        input_depth3 = np.uint8(input_depth3[0].detach().cpu().numpy()/input_depth3.max().item() * 255.0)
        scan_path = os.path.join(output_path, scan_name)
        if os.path.exists(scan_path) is False:
            os.makedirs(scan_path)
        imageio.imwrite(os.path.join(scan_path, "input_view1.png"), np.uint8(input_view1 * 255.0))
        imageio.imwrite(os.path.join(scan_path, "input_view2.png"), np.uint8(input_view2 * 255.0))
        imageio.imwrite(os.path.join(scan_path, "input_view3.png"), np.uint8(input_view3 * 255.0))

        imageio.imwrite(os.path.join(scan_path, "input_depth1.png"), input_depth1)
        imageio.imwrite(os.path.join(scan_path, "input_depth2.png"), input_depth2)
        imageio.imwrite(os.path.join(scan_path, "input_depth3.png"), input_depth3)

        
        target_view_list = np.ones(len(batch['images']))
        target_view_list[src_list] = 0
        target_view_list = np.nonzero(target_view_list)[0]
        target_view_list = target_view_list.tolist()

        ssim_pred_collect = []
        psnr_pred_collect = []
        ssim_pred_collect_exclude = []
        psnr_pred_collect_exclude = []
        gts = []
        preds = []
        gts_exclude = []
        preds_exclude = []
        for i in range(results[1]['OutputImg'].shape[0]):
            target_view = target_view_list[i]
            target = results[1]['OutputImg'][i]
            pred = results[1]['PredImg'][i]
            gts.append(target)
            preds.append(pred)

            target = tensor_to_image(target)
            pred = tensor_to_image(pred)
            imageio.imwrite(os.path.join(scan_path, "output_view_{0:06d}.png".format(target_view)), np.uint8(pred * 255.0))
            imageio.imwrite(os.path.join(scan_path, "target_view_{0:06d}.png".format(target_view)), np.uint8(target * 255.0))
            # We calculate the psnr and ssim
            psnr_pred = peak_signal_noise_ratio(target, pred, data_range=1)
            ssim_pred = structural_similarity(
                        target,
                        pred,
                        multichannel=True,
                        data_range=1)

            psnr_pred_collect.append(psnr_pred)
            ssim_pred_collect.append(ssim_pred)
            if target_view not in base_exclude_views:
                gts_exclude.append(results[1]['OutputImg'][i])
                preds_exclude.append(results[1]['PredImg'][i])
                psnr_pred_collect_exclude.append(psnr_pred)
                ssim_pred_collect_exclude.append(ssim_pred)

        gts = torch.stack(gts)
        preds = torch.stack(preds)
        preds_spl = torch.split(preds, args.lpips_batch_size, dim=0)
        gts_spl = torch.split(gts, args.lpips_batch_size, dim=0)
        lpips_all = []  
        with torch.no_grad():
            for predi, gti in zip(preds_spl, gts_spl):
                lpips_i = lpips_vgg(predi.to(device=device), gti.to(device=device))
                lpips_all.append(lpips_i)
            lpips_all = torch.cat(lpips_all)
        lpips_total = lpips_all.mean().item()
        lpips_all = []  
        gts_exclude = torch.stack(gts_exclude)
        preds_exclude = torch.stack(preds_exclude)
        preds_spl = torch.split(preds_exclude, args.lpips_batch_size, dim=0)
        gts_spl = torch.split(gts_exclude, args.lpips_batch_size, dim=0)
        with torch.no_grad():
            for predi, gti in zip(preds_spl, gts_spl):
                lpips_i = lpips_vgg(predi.to(device=device), gti.to(device=device))
                lpips_all.append(lpips_i)
            lpips_all = torch.cat(lpips_all)
        lpips_exclude = lpips_all.mean().item()
        evaluation[scan_name] = {}
        evaluation[scan_name].update({"all_pred_psnr": np.mean(psnr_pred_collect), "all_pred_ssim":np.mean(ssim_pred_collect), "all_pred_lpips":lpips_total, "exclu_pred_psnr":np.mean(psnr_pred_collect_exclude), "exclu_pred_ssim":np.mean(ssim_pred_collect_exclude), "excludee_pred_lpips":lpips_exclude, 'Depth_Error':Depth_error.item()})

    finish_file_name = os.path.join(outpath, name, "finish.txt")
    finish_file = open(finish_file_name, 'a',buffering=1)
    finish_file.write("-----------all dataset evaluation------------")
    pred_psnr_collect = []
    pred_ssim_collect = []
    pred_lpips_collect = []
    Depth_Error_collect =[]
    for scan in evaluation.keys():
        pred_psnr = evaluation[scan]['all_pred_psnr']
        pred_ssim = evaluation[scan]['all_pred_ssim']
        pred_lpips = evaluation[scan]['all_pred_lpips']
        depth_error = evaluation[scan]['Depth_Error']
        pred_psnr_collect.append(pred_psnr)
        pred_ssim_collect.append(pred_ssim)
        pred_lpips_collect.append(pred_lpips)
        Depth_Error_collect.append(depth_error)
        finish_file.write("{}: output psnr {}, output ssim {}, lpips {}, Depth_Error {}\n".format(scan, pred_psnr, pred_ssim, pred_lpips, depth_error))
    finish_file.write("Total: output psnr {}, output ssim {}, lpips{}, Depth Error {}\n".format( np.mean(pred_psnr_collect), np.mean(pred_ssim_collect), np.mean(pred_lpips_collect), np.mean(Depth_Error_collect)))

    finish_file.write("-----------excluded dataset evaluation-----------\n")
    pred_psnr_collect = []
    pred_ssim_collect = []
    pred_lpips_collect = []

    for scan in evaluation.keys():
        pred_psnr = evaluation[scan]['exclu_pred_psnr']
        pred_ssim = evaluation[scan]['exclu_pred_ssim']
        pred_lpips = evaluation[scan]['excludee_pred_lpips']
        pred_psnr_collect.append(pred_psnr)
        pred_ssim_collect.append(pred_ssim)
        pred_lpips_collect.append(pred_lpips)
        finish_file.write("{}: output psnr {}, output ssim {}, pred lpips {}\n".format(scan, pred_psnr, pred_ssim, pred_lpips))

    finish_file.write("Total: output psnr {}, output ssim {},pred lpips {}\n".format( np.mean(pred_psnr_collect), np.mean(pred_ssim_collect), np.mean(pred_lpips_collect)))
    finish_file.close()
    dict_name = os.path.join(outpath, name, "evaluation.pt")
    torch.save(evaluation, dict_name)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--name", type=str, default="results")
    parser.add_argument("--src_list", type=str, default='22 25 28')
    parser.add_argument("--input_view", type=int, default=3)
    parser.add_argument("--winsize", type=int, default=256)
    parser.add_argument("--lpips_batch_size", type=int, default=16)
    args = parser.parse_args()

    main(args)
 
