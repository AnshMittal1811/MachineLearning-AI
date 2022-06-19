# config
import sys
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from ptflops import get_model_complexity_info
from config import config
import net.networks as networks
from eval.Estimator_fast import Estimator
from options.train_options import TrainOptions
from Dataset.DatasetConstructor_fast import TrainDatasetConstructor,EvalDatasetConstructor
from PIL import Image
Image.MAX_IMAGE_PIXELS = None




if __name__ == '__main__':
    opt = TrainOptions().parse()

    # Mainly get settings for specific datasets
    setting = config(opt)

    log_file = os.path.join(setting.model_save_path, opt.dataset_name+'.log')
    log_f = open(log_file, "w")

    # Data loaders
    train_dataset = TrainDatasetConstructor(
        setting.train_num,
        setting.train_img_path,
        setting.train_gt_map_path,
        mode=setting.mode,
        dataset_name=setting.dataset_name,
        device=setting.device,
        is_random_hsi=setting.is_random_hsi,
        is_flip=setting.is_flip,
        fine_size=opt.fine_size
        )
    eval_dataset = EvalDatasetConstructor(
        setting.eval_num,
        setting.eval_img_path,
        setting.eval_gt_map_path,
        mode=setting.mode,
        dataset_name=setting.dataset_name,
        device=setting.device)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=setting.batch_size, num_workers=opt.nThreads, drop_last=True)


    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)

    # model construct
    net = networks.define_net(opt)
    net = networks.init_net(net, gpu_ids=opt.gpu_ids)
    if opt.continue_train:
        net.module.load_state_dict(torch.load(opt.model_name, map_location=str(setting.device)))



    criterion = nn.MSELoss(reduction='sum').to(setting.device) # first device is ok
    offset_crit = nn.MSELoss(reduction='sum').to(setting.device) # first device is ok
    estimator = Estimator(opt, setting, eval_loader, criterion=criterion)

    optimizer = networks.select_optim(net, opt)

    def cal_scale_rate(cur_epoch, start_decay_epoch, epoch_decay_step, decay_rate):
        if cur_epoch <= start_decay_epoch:
            return 1
        else:
            return decay_rate**((cur_epoch - start_decay_epoch)//epoch_decay_step + 1)



    step = 0
    eval_loss, eval_mae, eval_rmse = [], [], []

    base_mae = float(opt.base_mae)

    for epoch_index in range(setting.epoch):
        # eval
        if epoch_index % opt.eval_per_epoch == 0 and epoch_index > opt.start_eval_epoch:
            print('Evaluating step:', str(step), '\t epoch:', str(epoch_index))
            validate_MAE, validate_RMSE, validate_loss, time_cost = estimator.evaluate(net, False) # pred_mae and pred_mse are for seperate datasets
            eval_loss.append(validate_loss)
            eval_mae.append(validate_MAE)
            eval_rmse.append(validate_RMSE)
            log_f.write(
                'In step {}, epoch {}, loss = {}, mae = {}, mse = {}, time cost eval = {}s\n'
                .format(step, epoch_index, validate_loss, validate_MAE, validate_RMSE, time_cost))
            log_f.flush()
            # save model with epoch and MAE

            # Two kinds of conditions, we save models
            save_now = False

            if validate_MAE < base_mae:
                save_now = True

            if save_now:
                best_model_name = setting.model_save_path + "/MAE_" + str(round(validate_MAE, 2)) + \
                    "_MSE_" + str(round(validate_RMSE, 2)) + '_Ep_' + str(epoch_index) + '.pth'
                if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), best_model_name)
                    net.cuda(opt.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), best_model_name)


        time_per_epoch = 0
        for train_img, train_gt in train_loader:
            # put data to setting.device
            train_img = train_img.to(setting.device)
            train_gt = train_gt.to(setting.device)

            net.train()
            x, y = train_img, train_gt
            start = time.time()
            if (opt.extra_loss):
                prediction, x_offset_list= net(x, out_feat=True)
            else:
                prediction = net(x)

            extra_loss = 0

            if opt.extra_loss:
                target_offset = torch.zeros_like(x_offset_list[0][:, 0, :,:]) # c=1

                for (x_offset) in zip(x_offset_list):
                    if opt.extra_loss:
                        '''
                        A   B   C
                        D   E   F
                        G   H   I
                        '''
                        # 9 points: stored in `_h, _w` layout
                        r = 2                    
                        A_y, A_x = x_offset[:, 0, :, :], x_offset[:, 1, :, :]
                        B_y, B_x = x_offset[:, 2, :, :], x_offset[:, 3, :, :]
                        C_y, C_x = x_offset[:, 4, :, :], x_offset[:, 5, :, :]
                        D_y, D_x = x_offset[:, 6, :, :], x_offset[:, 7, :, :]
                        E_y, E_x = x_offset[:, 8, :, :], x_offset[:, 9, :, :]
                        F_y, F_x = x_offset[:, 10, :, :], x_offset[:, 11, :, :]
                        G_y, G_x = x_offset[:, 12, :, :], x_offset[:, 13, :, :]
                        H_y, H_x = x_offset[:, 14, :, :], x_offset[:, 15, :, :]
                        I_y, I_x = x_offset[:, 16, :, :], x_offset[:, 17, :, :]
                        # for up-down direction (using (0,1) or (0, -1), it's all ok)
                        extra_loss += offset_crit(B_y + H_y, 2*E_y.detach())
                        # extra_loss += offset_crit(B_x, E_x.detach())
                        extra_loss += offset_crit((B_x + H_x), 2*E_x.detach())
                        # for left-right direction (using (1,0) or (-1,0), it's all ok)
                        extra_loss += offset_crit(D_x + F_x, 2*E_x.detach())
                        extra_loss += offset_crit((D_y + F_y), 2*E_y.detach())
                        # extra_loss += offset_crit(F_y, E_y.detach())
                        # for center pos
                        extra_loss += offset_crit(E_x, target_offset.detach())
                        extra_loss += offset_crit(E_y, target_offset.detach())

                        # simpler constrain, letting ADG in the line, same as ABC, CFI, GHI.

                        A_x_ori, A_y_ori = - r, - r
                        B_x_est, B_y_est = B_x, B_y - r
                        C_x_ori, C_y_ori = r, -r
                        D_x_est, D_y_est = D_x - r, D_y
                        E_x_est, E_y_est = E_x, E_y
                        F_x_est, F_y_est = F_x + r, F_y
                        G_x_ori, G_y_ori = - r, r
                        H_x_est, H_y_est = H_x, H_y + r
                        I_x_ori, I_y_ori = r, r                                    

                        extra_loss += offset_crit(A_x + G_x, 2*D_x.detach())
                        extra_loss += offset_crit(A_y + G_y, 2*D_y.detach())
                        extra_loss += offset_crit(A_x + C_x, 2*B_x.detach())
                        extra_loss += offset_crit(A_y + C_y, 2*B_y.detach())
                        extra_loss += offset_crit(C_x + I_x, 2*F_x.detach())
                        extra_loss += offset_crit(C_y + I_y, 2*F_y.detach())
                        extra_loss += offset_crit(I_x + G_x, 2*H_x.detach())
                        extra_loss += offset_crit(I_y + G_y, 2*H_y.detach())
                    
            extra_loss *= opt.extra_w
            loss = criterion(prediction, y)
            optimizer.zero_grad()
            (loss + extra_loss).backward()
            loss_item = loss.detach().item()
            loss_extra_item = extra_loss.item() if opt.extra_loss else 0
            optimizer.step()

            step += 1
            end = time.time()
            time_per_epoch += end - start

            if step % opt.print_step == 0:
                print("Step:{:d}\t, Epoch:{:d}\t, Loss:{:.4f}, Extra_loss:{:.4f}".format(step, epoch_index, loss_item, loss_extra_item))


    net =models.vgg16_bn()
    flops, params =get_model_complexity_info(net, (3,224,224), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('Flops:{}'.format(flops))
    print('Params:'+ params)
