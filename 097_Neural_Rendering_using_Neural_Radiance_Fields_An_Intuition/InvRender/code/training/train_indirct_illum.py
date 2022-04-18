import os
import sys
from datetime import datetime

import imageio
import numpy as np
import torch
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
from model.loss import query_indir_illum

class IllumTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = kwargs['exps_folder_name']
        self.batch_size = kwargs['batch_size']
        self.nepochs = self.conf.get_int('train.illum_epoch')
        self.max_niters = kwargs['max_niters']
        self.GPU_INDEX = kwargs['gpu_index']

        self.expname = 'Illum-' + kwargs['expname']
        
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.illum_optimizer_params_subdir = "IllumOptimizerParameters"
        self.illum_scheduler_params_subdir = "IllumSchedulerParameters"
        self.vis_optimizer_params_subdir = "VisOptimizerParameters"
        self.vis_scheduler_params_subdir = "VisSchedulerParameters"
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.illum_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.illum_scheduler_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.vis_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.vis_scheduler_params_subdir))

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))
        
        print('Loading data ...')
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(
                                kwargs['data_split_dir'], kwargs['frame_skip'], split='train')
        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=1,
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )
        
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        if torch.cuda.is_available():
            self.model.cuda()

        self.illum_loss = utils.get_class(self.conf.get_string('train.illum_loss_class'))(**self.conf.get_config('illum_loss'))
        self.illum_optimizer = torch.optim.Adam(self.model.indirect_illum_network.parameters(),
                                                lr=self.conf.get_float('train.illum_learning_rate'))
        self.illum_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.illum_optimizer,
                                                self.conf.get_list('train.illum_sched_milestones', default=[]),
                                                gamma=self.conf.get_float('train.illum_sched_factor', default=0.0))
        self.vis_optimizer = torch.optim.Adam(self.model.visibility_network.parameters(),
                                                lr=self.conf.get_float('train.illum_learning_rate'))
        self.vis_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.vis_optimizer,
                                                self.conf.get_list('train.illum_sched_milestones', default=[]),
                                                gamma=self.conf.get_float('train.illum_sched_factor', default=0.0))

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            print('Loading pretrained model: ', os.path.join(
                    old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.illum_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.illum_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.illum_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.illum_scheduler.load_state_dict(data["scheduler_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.vis_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.vis_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.vis_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.vis_scheduler.load_state_dict(data["scheduler_state_dict"])

        geo_dir = os.path.join('../',kwargs['exps_folder_name'], 'IDR-' + kwargs['expname'])
        if os.path.exists(geo_dir):
            timestamps = os.listdir(geo_dir)
            timestamp = sorted(timestamps)[-1] # using the newest training result
        else:
            print('No geometry pretrain, please train idr first!')
            exit(0)

        # reloading geometry and radiance
        geo_path = os.path.join(geo_dir, timestamp) + '/checkpoints/ModelParameters/latest.pth'
        print('Reloading geometry from: ', geo_path)
        model = torch.load(geo_path)['model_state_dict']
        geometry = {k.split('network.')[1]: v for k, v in model.items() if 'implicit_network' in k}
        radiance = {k.split('network.')[1]: v for k, v in model.items() if 'rendering_network' in k}
        self.model.implicit_network.load_state_dict(geometry)
        self.model.rendering_network.load_state_dict(radiance)

        self.num_pixels = self.conf.get_int('train.illum_num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')


    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.illum_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.illum_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.illum_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.illum_optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.illum_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.illum_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.illum_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.illum_scheduler_params_subdir, "latest.pth"))
        
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.vis_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.vis_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.vis_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.vis_optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.vis_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.vis_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.vis_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.vis_scheduler_params_subdir, "latest.pth"))


    def plot_to_disk(self):
        self.model.eval()
        sampling_idx = self.train_dataset.sampling_idx
        self.train_dataset.change_sampling_idx(-1)
        indices, model_input, ground_truth = next(iter(self.plot_dataloader))

        for key in model_input.keys():
            model_input[key] = model_input[key].cuda()

        split = utils.split_input(model_input, self.total_pixels)
        res = []
        for s in split:
            out = self.model(s, trainstage="Illum")
            trace_outputs = self.model.trace_radiance(out, nsamp=4)

            points_mask = out["network_object_mask"]
            trace_radiance = torch.mean(trace_outputs['trace_radiance'], axis=1)
            pred_radiance = torch.zeros_like(trace_outputs['trace_radiance']).cuda()
            if points_mask.sum() > 0:
                pred_radiance[points_mask] = query_indir_illum(
                    out["indirect_sgs"][points_mask].detach(), trace_outputs['sample_dirs'])
            pred_radiance = torch.mean(pred_radiance, axis=1)

            _, pred_vis = torch.max(trace_outputs["pred_vis"].detach(), dim=-1)
            pred_vis = torch.mean(pred_vis.float(), axis=1)
            gt_vis = torch.mean((~trace_outputs["gt_vis"]).float(), axis=1)[:, 0]
            
            res.append({
                'points': out['points'],
                'network_object_mask': points_mask,
                'traced_radiance': trace_radiance,
                'pred_radiance': pred_radiance,
                'pred_vis': pred_vis,
                'gt_vis': gt_vis,
            })

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

        plt.plot_illum(
                 model_outputs,
                 ground_truth['rgb'],
                 self.plots_dir,
                 self.cur_iter,
                 self.img_res,
                 )
        self.model.train()
        self.train_dataset.sampling_idx = sampling_idx

    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)

        for epoch in range(self.start_epoch, self.nepochs + 1):
            self.train_dataset.change_sampling_idx(self.num_pixels)

            if self.cur_iter > self.max_niters:
                self.save_checkpoints(epoch)
                self.plot_to_disk()
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                if self.cur_iter % self.ckpt_freq == 0:
                    self.save_checkpoints(epoch)

                if self.cur_iter % self.plot_freq == 0:
                    self.plot_to_disk()

                for key in model_input.keys():
                    model_input[key] = model_input[key].cuda()

                model_outputs = self.model(model_input, trainstage='Illum')
                trace_outputs = self.model.trace_radiance(model_outputs, nsamp=16)
                radiance_loss, visibility_loss = self.illum_loss(model_outputs, trace_outputs)

                # update vis
                self.vis_optimizer.zero_grad()
                visibility_loss.backward()
                self.vis_optimizer.step()

                # update illum
                self.illum_optimizer.zero_grad()
                radiance_loss.backward()
                self.illum_optimizer.step()

                if self.cur_iter % 50 == 0:
                    print('{0} [{1}] ({2}/{3}): radiance_loss = {4}, visibility_loss = {5}'
                            .format(self.expname, epoch, data_index, self.n_batches, 
                                    radiance_loss.item(), visibility_loss.item()))
                    self.writer.add_scalar('radiance_loss', radiance_loss.item(), self.cur_iter)
                    self.writer.add_scalar('visibility_loss', visibility_loss.item(), self.cur_iter)

                self.cur_iter += 1
                self.illum_scheduler.step()
                self.vis_scheduler.step()
