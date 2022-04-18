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
from model.sg_render import compute_envmap


class MaterialTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.exps_folder_name = kwargs['exps_folder_name']
        self.batch_size = kwargs['batch_size']
        self.nepochs =self.conf.get_int('train.sg_epoch')
        self.max_niters = kwargs['max_niters']
        self.GPU_INDEX = kwargs['gpu_index']

        self.expname = 'Mat-' + kwargs['expname']

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
        self.sg_optimizer_params_subdir = "SGOptimizerParameters"
        self.sg_scheduler_params_subdir = "SGSchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir))

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

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))
        self.sg_optimizer = torch.optim.Adam(self.model.envmap_material_network.parameters(),
                                                lr=self.conf.get_float('train.sg_learning_rate'))
        self.sg_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.sg_optimizer,
                                                self.conf.get_list('train.sg_sched_milestones', default=[]),
                                                gamma=self.conf.get_float('train.sg_sched_factor', default=0.0))

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
                os.path.join(old_checkpnts_dir, self.sg_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.sg_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.sg_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.sg_scheduler.load_state_dict(data["scheduler_state_dict"])

        illum_dir = os.path.join('../',kwargs['exps_folder_name'], 'Illum-' + kwargs['expname'])
        if os.path.exists(illum_dir):
            timestamps = os.listdir(illum_dir)
            timestamp = sorted(timestamps)[-1] # using the newest training result
        else:
            print('No illum_model pretrain, please train it first!')
            exit(0)

        # reload pretrain geometry model & indirect illumination model
        illum_path = os.path.join(illum_dir, timestamp) + '/checkpoints/ModelParameters/latest.pth'
        print('Reloading indirect illumination from: ', illum_path)
        model = torch.load(illum_path)['model_state_dict']

        geometry = {k.split('network.')[1]: v for k, v in model.items() if 'implicit_network' in k}
        radiance = {k.split('network.')[1]: v for k, v in model.items() if 'rendering_network' in k}
        self.model.implicit_network.load_state_dict(geometry)
        self.model.rendering_network.load_state_dict(radiance)

        incident_radiance = {k.split('network.')[1]: v for k, v in model.items() if 'indirect_illum_network' in k}
        visibility = {k.split('network.')[1]: v for k, v in model.items() if 'visibility_network' in k}
        self.model.indirect_illum_network.load_state_dict(incident_radiance)
        self.model.visibility_network.load_state_dict(visibility)

        self.num_pixels = self.conf.get_int('train.num_pixels')
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
            {"epoch": epoch, "optimizer_state_dict": self.sg_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.sg_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.sg_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.sg_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, "latest.pth"))

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
            out = self.model(s, trainstage="Material")
            res.append({
                'points': out['points'].detach(),
                'normals': out['normals'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'object_mask': out['object_mask'].detach(),
                'roughness':out['roughness'].detach(),
                'diffuse_albedo': out['diffuse_albedo'].detach(),
                'sg_specular_rgb': out['sg_specular_rgb'].detach(),
                'indir_rgb':out['indir_rgb'].detach(),
                'sg_rgb': out['sg_rgb'].detach() + out['indir_rgb'].detach(),
                'vis_shadow': out['vis_shadow'].detach(),
            })

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

        plt.plot_mat(
                 model_outputs,
                 ground_truth['rgb'],
                 self.plots_dir,
                 self.cur_iter,
                 self.img_res,
                 )

        # log environment map
        lgtSGs = self.model.envmap_material_network.get_light()
        envmap = compute_envmap(lgtSGs=lgtSGs, 
                        H=256, W=512, upper_hemi=self.model.envmap_material_network.upper_hemi)
        envmap = envmap.cpu().numpy()
        imageio.imwrite(os.path.join(self.plots_dir, 'envmap1_{}.exr'.format(self.cur_iter)), envmap)

        self.model.train()
        self.train_dataset.sampling_idx = sampling_idx

    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)
        mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)

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

                model_outputs = self.model(model_input, trainstage="Material")
                loss_output = self.loss(model_outputs, ground_truth, 
                                mat_model=self.model.envmap_material_network,train_idr=False)

                loss = loss_output['loss']

                self.sg_optimizer.zero_grad()
                loss.backward()
                self.sg_optimizer.step()

                if self.cur_iter % 50 == 0:
                    print('{0} [{1}] ({2}/{3}): loss = {4}, '
                        'sg_rgb_loss = {5}, sg_lr = {6}, sg_psnr = {7}, kl_loss={8}, latent_smooth_loss={9}'
                            .format(self.expname, epoch, indices, self.n_batches, 
                                    loss.item(),
                                    loss_output['sg_rgb_loss'].item(),
                                    self.sg_scheduler.get_last_lr()[0],
                                    mse2psnr(loss_output['sg_rgb_loss'].item()),
                                    loss_output['kl_loss'].item(),
                                    loss_output['latent_smooth_loss'].item()))

                    self.writer.add_scalar('sg_rgb_loss', loss_output['sg_rgb_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('sg_psnr', mse2psnr(loss_output['sg_rgb_loss'].item()), self.cur_iter)
                    self.writer.add_scalar('kl_loss', loss_output['kl_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('sg_lrate', self.sg_scheduler.get_last_lr()[0], self.cur_iter)

                self.cur_iter += 1
                self.sg_scheduler.step()
