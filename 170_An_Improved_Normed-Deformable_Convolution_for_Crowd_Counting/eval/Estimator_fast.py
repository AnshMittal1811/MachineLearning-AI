import random
import math
import copy
import numpy as np
import os
import sys
from PIL import Image
from metrics import AEBatch, SEBatch
import time
import torch
import scipy.io as scio
import cv2
class Estimator(object):
    def __init__(self, opt, setting, eval_loader, criterion=torch.nn.MSELoss(reduction="sum")):
        self.setting = setting
        self.ae_batch = AEBatch().to(self.setting.device)
        self.se_batch = SEBatch().to(self.setting.device)
        self.criterion = criterion
        self.dataset_name = self.setting.dataset_name
        self.opt = opt
        self.eval_loader = eval_loader

    def evaluate(self, net, is_show=False):
        net.eval()
        MAE_, MSE_, loss_ = [], [], []
        rand_number, cur, time_cost = random.randint(0, self.setting.eval_num - 1), 0, 0
        for eval_img_path, eval_img, eval_gt, ph_min, pw_min, idx_h, idx_w in self.eval_loader:
            eval_img = eval_img.to(self.setting.device)
            eval_gt = eval_gt.to(self.setting.device)

            start = time.time()
            eval_patchs = torch.squeeze(eval_img)
            eval_gt_shape = eval_gt.shape
            prediction_map = torch.zeros_like(eval_gt)
            eval_img_path = eval_img_path[0]

            with torch.no_grad():
                eval_prediction = net(eval_patchs)
                eval_patchs_shape = eval_prediction.shape
                # test cropped patches
                self.test_crops(eval_patchs_shape, eval_prediction, prediction_map)
                # only for OUC_dataset
                if self.dataset_name.find('OUC') != -1:
                    gt_counts = self.get_gt_num_for_ouc(eval_img_path)
                else:
                    gt_counts = self.get_gt_num(eval_img_path)
                # calculate metrics
                batch_ae = self.ae_batch(prediction_map, gt_counts).data.cpu().numpy()
                batch_se = self.se_batch(prediction_map, gt_counts).data.cpu().numpy()

                loss = self.criterion(prediction_map, eval_gt)
                loss_.append(loss.data.item())
                MAE_.append(batch_ae)
                MSE_.append(batch_se)

            cur += 1
            torch.cuda.synchronize()
            end = time.time()
            time_cost += (end - start)
            vis_img = prediction_map[0, 0].cpu().numpy()
        # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            cv2.imwrite("/home/weigang1/zx/hrnet_counting-master_QNRF/OUC_Crowd_den/predict/"+os.path.basename(eval_img_path)[:-4]+'.png', vis_img)
        # return the validate loss, validate MAE and validate RMSE
        MAE_, MSE_, loss_ = np.reshape(MAE_, [-1]), np.reshape(MSE_, [-1]), np.reshape(loss_, [-1])
        return np.mean(MAE_), np.sqrt(np.mean(MSE_)), np.mean(loss_), time_cost

            #vis_img = prediction_map[0, 0].cpu().numpy()
        # normalize density map values from 0 to 1, then map it to 0-255.
            #vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            #vis_img = (vis_img * 255).astype(np.uint8)
            #vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            #cv2.imwrite("/home/weigang1/zx/hrnet_counting-master_QNRF/output/predict/"+os.path.basename(eval_img_path)[:-4]+'.png', vis_img)
            #cur += 1
            #torch.cuda.synchronize()
            #end = time.time()
            #time_cost += (end - start)

        # return the validate loss, validate MAE and validate RMSE
        #MAE_, MSE_, loss_ = np.reshape(MAE_, [-1]), np.reshape(MSE_, [-1]), np.reshape(loss_, [-1])
        #return np.mean(MAE_), np.sqrt(np.mean(MSE_)), np.mean(loss_), time_cost

    # New Function
    def get_gt_num(self, eval_img_path):
        mat_name = eval_img_path.replace('images', 'ground_truth')[:-4] + ".mat"
        gt_counts = len(scio.loadmat(mat_name)['annPoints'])

        return gt_counts

    def get_gt_num_for_ouc(self, eval_img_path):
        tmp_name = os.path.splitext(os.path.basename(eval_img_path))[0].replace('IMG', 'GT_IMG') + ".npy"
        mat_name = os.path.split(eval_img_path)[0].replace('images', 'ground_truth') + '/' + tmp_name
        info = np.load(mat_name,allow_pickle=True)
        gt_counts = len(info)
        return gt_counts

    def test_crops(self, eval_shape, eval_p, pred_m):
        for i in range(3):
            for j in range(3):
                start_h, start_w = math.floor(eval_shape[2] / 4), math.floor(eval_shape[3] / 4)
                valid_h, valid_w = eval_shape[2] // 2, eval_shape[3] // 2
                pred_h = math.floor(3 * eval_shape[2] / 4) + (eval_shape[2] // 2) * (i - 1)
                pred_w = math.floor(3 * eval_shape[3] / 4) + (eval_shape[3] // 2) * (j - 1)
                if i == 0:
                    valid_h = math.floor(3 * eval_shape[2] / 4)
                    start_h = 0
                    pred_h = 0
                elif i == 2:
                    valid_h = math.ceil(3 * eval_shape[2] / 4)

                if j == 0:
                    valid_w = math.floor(3 * eval_shape[3] / 4)
                    start_w = 0
                    pred_w = 0
                elif j == 2:
                    valid_w = math.ceil(3 * eval_shape[3] / 4)
                pred_m[:, :, pred_h:pred_h + valid_h, pred_w:pred_w + valid_w] += eval_p[i * 3 + j:i * 3 + j + 1, :,start_h:start_h + valid_h, start_w:start_w + valid_w]
