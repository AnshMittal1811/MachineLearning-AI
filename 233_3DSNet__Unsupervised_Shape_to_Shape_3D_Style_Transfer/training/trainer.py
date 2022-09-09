"""
Copyright (c) 2021, Mattia Segu
Licensed under the MIT License (see LICENSE for details)
"""

import torch
import os
import shutil
import auxiliary.html_report as html_report
import numpy as np
from easydict import EasyDict
import pymesh
from termcolor import colored
import auxiliary.my_utils as my_utils
# import pymeshlab as ml

from training.trainer_abstract import TrainerAbstract
import dataset.mesh_processor as mesh_processor
from training.trainer_iteration import TrainerIteration
from model.trainer_model import TrainerModel
from dataset.trainer_dataset import TrainerDataset
from training.trainer_loss import TrainerLoss


class Trainer(TrainerAbstract, TrainerLoss, TrainerIteration, TrainerDataset, TrainerModel):
    def __init__(self, opt):
        """
        Main Trainer class inheriting from the other main modules.
        It implements all functions related to train and evaluate for an epoch.
        """

        super(Trainer, self).__init__(opt)
        self.dataset_train = None
        self.num_interpolations = self.opt.num_interpolations
        self.opt.training_media_path = os.path.join(self.opt.dir_name, "training_media")
        self.opt.demo_media_path = os.path.join(self.opt.dir_name, "demo_media")
        if not os.path.exists(self.opt.demo_media_path):
            os.mkdir(self.opt.demo_media_path)
        if not os.path.exists(self.opt.training_media_path):
            os.mkdir(self.opt.training_media_path)

        # Define Flags
        self.flags = EasyDict()
        self.flags.media_count = 0
        self.flags.add_log = True
        self.flags.build_website = False
        self.flags.get_closer_neighbourg = False
        self.flags.compute_clustering_errors = False
        self.display = EasyDict({"recons": []})
        self.colormap = mesh_processor.ColorMap()
        self.mesh_ext = 'ply' if self.opt.decoder_type.lower() == 'atlasnet' else 'obj'

    def train_loop(self):
        """
        Do a single pass on all training data
        """
        for _, (data_a, data_b) in enumerate(zip(self.datasets.dataloader_train[self.classes[0]],
                                                 self.datasets.dataloader_train[self.classes[1]])):
            self.increment_iteration()
            data_a = EasyDict(data_a)
            data_b = EasyDict(data_b)
            data_a.points = data_a.points.to(self.opt.device)
            data_b.points = data_b.points.to(self.opt.device)
            if len(data_a.points) == len(data_b.points):
                if self.datasets.data_augmenter is not None:
                    # Apply data augmentation on points (nothing for now)
                    self.datasets.data_augmenter(data_a.points)
                    self.datasets.data_augmenter(data_b.points)
                self.train_iteration(data_a, data_b)

    def train_epoch(self):
        """ Launch a training epoch """
        self.flags.train = True
        if self.epoch == (self.opt.nepoch - 1):
            # Flag last epoch
            self.flags.build_website = True

        self.log.reset()
        if not self.opt.no_learning:
            self.network.train()
        else:
            self.network.eval()
        self.learning_rate_scheduler()
        self.reset_iteration()
        for i in range(self.opt.loop_per_epoch):
            self.train_loop()

    def test_loop(self):
        """
        Do a single pass on all test data
        """
        self.reset_iteration()
        for _, (data_a, data_b) in enumerate(zip(self.datasets.dataloader_test[self.classes[0]],
                                                 self.datasets.dataloader_test[self.classes[1]])):
            self.increment_iteration()
            data_a = EasyDict(data_a)
            data_b = EasyDict(data_b)
            data_a.points = data_a.points.to(self.opt.device)
            data_b.points = data_b.points.to(self.opt.device)

            if len(data_a.points) == len(data_b.points):
                self.test_iteration(data_a, data_b)

    def test_epoch(self):
        """ Launch an test epoch """
        self.flags.train = False
        self.network.eval()

        # Evaluation of average LPIPS distance between randomly generated pairs with different styles
        # (the higher the more various are the generated samples)
        if (self.flags.build_website or self.opt.run_single_eval) and not self.opt.no_lpips:
            my_utils.cyan_print(f"Evaluating average LPIPS on {self.opt.num_samples} samples per domain, "
                  f"each with {self.opt.num_pairs} different style pairs. It will take a while...")
            my_utils.yellow_print(f"Noise magnitude = {self.opt.noise_magnitude}")
            self.lpips_dict = self.evaluate_average_lpips(num_inputs=self.opt.num_samples, num_pairs=self.opt.num_pairs)
            self.dump_lpips()

        if not self.opt.no_quantitative_eval:
            self.test_loop()
            self.log.end_epoch()
            print(f"Sampled {self.num_val_points} regular points for evaluation")

        try:
            self.log.update_curves(self.visualizer.vis, self.opt.dir_name)
        except:
            print("could not update curves")

        self.metro_results = 0
        if (self.flags.build_website or self.opt.run_single_eval) and not self.opt.no_metro:
            self.metro()

        if self.flags.build_website:
            # Build report using Netvision.
            self.html_report_data = EasyDict()
            self.html_report_data.output_meshes = [self.generate_random_mesh() for i in range(10)]
            if not self.opt.no_quantitative_eval:
                if self.opt.run_single_eval:
                    log_curves = ["loss_val"]
                else:
                    log_curves = ["loss_val", "loss_train_gen", "loss_train_dis"]
                self.html_report_data.data_curve = {key: [np.log(val) for val in self.log.curves[key]] for key in
                                                    log_curves}
                self.html_report_data.fscore_curve = {"fscore": self.log.curves["fscore"]}
                self.html_report_data.chamfer_distance_curve = {"chamfer_distance": self.log.curves["chamfer_distance"]}
                if "lpips" in self.log.curves:
                    self.html_report_data.average_lpips = {"lpips": self.log.curves["lpips"]}
                if "lpips_rec_from_source" in self.log.curves:
                    self.html_report_data.lpips_rec_from_source = {
                        "lpips_rec_from_source": self.log.curves["lpips_rec_from_source"]}
                if "lpips_rec_from_target" in self.log.curves:
                    self.html_report_data.lpips_rec_from_target = {
                        "lpips_rec_from_target": self.log.curves["lpips_rec_from_target"]}
                if "lpips_from_source" in self.log.curves:
                    self.html_report_data.lpips_from_source = {"lpips_from_source": self.log.curves["lpips_from_source"]}
                if "lpips_from_target" in self.log.curves:
                    self.html_report_data.lpips_from_target = {"lpips_from_target": self.log.curves["lpips_from_target"]}
                if "delta_source" in self.log.curves:
                    self.html_report_data.delta_source = {"delta_source": self.log.curves["delta_source"]}
                if "delta_target" in self.log.curves:
                    self.html_report_data.delta_target = {"delta_target": self.log.curves["delta_target"]}
                if "style_transfer_score" in self.log.curves:
                    self.html_report_data.style_transfer_score = {"style_transfer_score": self.log.curves["style_transfer_score"]}

                html_report.main(self, outHtml="index.html")

    def generate_random_mesh(self):
        """ Generate a mesh from a random test sample """
        index_a = np.random.randint(self.datasets.len_dataset_test[self.classes[0]])
        data_a = EasyDict(self.datasets.dataset_test[self.classes[0]][index_a])
        index_b = np.random.randint(self.datasets.len_dataset_test[self.classes[1]])
        data_b = EasyDict(self.datasets.dataset_test[self.classes[1]][index_b])

        data_a = EasyDict(self.datasets.dataset_test[self.classes[0]].load(data_a.pointcloud_path))
        data_b = EasyDict(self.datasets.dataset_test[self.classes[1]].load(data_b.pointcloud_path))
        # data_a.points.unsqueeze_(0)
        # data_b.points.unsqueeze_(0)
        return self.generate_mesh(data_a, data_b)

    def generate_mesh(self, data_a, data_b, save=True):
        """
        Generate a mesh from data and saves it.
        """
        self.make_network_input(data_a, self.SVR_0)
        self.make_network_input(data_b, self.SVR_1)
        x = {self.classes[0]: data_a.network_input,
             self.classes[1]: data_b.network_input}
        # Get normalization ops
        self.set_operation(data_a, data_b)
        path_a, path_b = self.copy_input_meshes(data_a, data_b, self.opt.training_media_path, save)

        # Get results of forward pass
        path_aa, _ = self.generate_mesh_from_classes(x, self.classes[0], self.classes[0], self.operation_a, save=save)
        path_ab, _ = self.generate_mesh_from_classes(x, self.classes[0], self.classes[1], self.operation_a, save=save)
        path_bb, _ = self.generate_mesh_from_classes(x, self.classes[1], self.classes[1], self.operation_b, save=save)
        path_ba, _ = self.generate_mesh_from_classes(x, self.classes[1], self.classes[0], self.operation_b, save=save)

        self.flags.media_count += 1
        return {"path_a": path_a, "path_b": path_b, "path_aa": path_aa, "path_bb": path_bb,
                "path_ba": path_ba, "path_ab": path_ab}

    def generate_mesh_from_classes(self, x, content_class, style_class, operation=None, demo=False, save=True):
        import time
        start = time.time()
        mesh = self.network.module.generate_mesh(x, content_class, style_class)

        base_dir = self.demo_pair_path if demo else self.opt.training_media_path
        path = '/'.join([base_dir, str(self.flags.media_count)]) + f"_{content_class}_{style_class}.ply"
        if save:
            self.save_mesh(mesh, rename_path(path, unnormalized=False, demo=demo, ext=self.mesh_ext), None)
            print(f"Saved generated mesh {rename_path(path, unnormalized=False, demo=demo, ext=self.mesh_ext)}! It took {time.time()-start} s")
            if operation and self.opt.decoder_type.lower() == 'atlasnet':  # TODO(msegu): make it work also for emshflow
                unnormalized_mesh = unnormalize(mesh, operation)
                self.save_mesh(unnormalized_mesh, rename_path(path, unnormalized=True, demo=demo, ext=self.mesh_ext), operation)
                print(f"Saved generated mesh {rename_path(path, unnormalized=True, demo=demo, ext=self.mesh_ext)}!")

        return rename_path(path, unnormalized=False, demo=demo, ext=self.mesh_ext), mesh

    def save_mesh(self, mesh, path, operation=None):
        base_dir = '/'.join(path.split('/')[:-1])
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if self.opt.decoder_type.lower() == 'atlasnet':
            if operation:
                unnormalized_mesh = unnormalize(mesh, operation)
                for name in mesh.get_attribute_names():
                    val = mesh.get_attribute(name)
                    unnormalized_mesh.add_attribute(name)
                    unnormalized_mesh.set_attribute(name, val)
                mesh = unnormalized_mesh
            mesh_processor.save(mesh, path, self.colormap)
        elif self.opt.decoder_type.lower() == 'meshflow':
            # mesh_processor.save(mesh, path, self.colormap)
            mesh.export(path)
        print(f"Saved generated mesh {path}!")

    def copy_input_meshes(self, data_a, data_b, base_dir, save=True):
        # Get inputs path
        path_a = data_a.path
        path_b = data_b.path
        path_a = pointcloud_to_shapenet_path(path_a)
        path_b = pointcloud_to_shapenet_path(path_b)
        # Get normalization ops
        self.set_operation(data_a, data_b)
        # Save a copy of input meshes
        if save and os.path.exists(path_a) and os.path.exists(path_b):
            copy_path_a = '/'.join([base_dir, str(self.flags.media_count)]) + \
                          f"_input_{self.classes[0]}.obj"
            copy_path_b = '/'.join([base_dir, str(self.flags.media_count)]) + \
                          f"_input_{self.classes[1]}.obj"
            shutil.copyfile(path_a, copy_path_a)
            shutil.copyfile(path_b, copy_path_b)
        return path_a, path_b

    def copy_input_renderings(self, data_a, data_b, base_dir, save=True):
        # Get inputs path
        path_a = data_a.path
        path_b = data_b.path
        path_a = pointcloud_to_renderings_path(path_a)
        path_b = pointcloud_to_renderings_path(path_b)
        # Get normalization ops
        self.set_operation(data_a, data_b)
        # Save a copy of input meshes
        if save and os.path.exists(path_a) and os.path.exists(path_b):
            copy_path_a = '/'.join([base_dir, 'rendering_' + str(self.flags.media_count)]) + \
                          f"_input_{self.classes[0]}"
            copy_path_b = '/'.join([base_dir, 'rendering_' + str(self.flags.media_count)]) + \
                          f"_input_{self.classes[1]}"
            shutil.copytree(path_a, copy_path_a)
            shutil.copytree(path_b, copy_path_b)
        return path_a, path_b

    def generate_mesh_interpolations(self, x):
        content_a, style_a = self.network.module.get_latent_codes(x, self.classes[0])
        content_b, style_b = self.network.module.get_latent_codes(x, self.classes[1])
        if self.opt.share_content_encoder:
            # You can interpolate across different styles and contents
            # Interpolate contents
            my_utils.magenta_print("Content interpolation...")
            contents_ab = np.linspace(content_a.cpu(), content_b.cpu(), self.num_interpolations)
            contents_ab = [torch.from_numpy(c).to(self.opt.device) for c in contents_ab]
            contents_ba = np.linspace(content_b.cpu(), content_a.cpu(), self.num_interpolations)
            contents_ba = [torch.from_numpy(c).to(self.opt.device) for c in contents_ba]
            # save meshes style_a
            print(f'Computing meshes for style {self.classes[0]}')
            out_a = [self.network.module.generate_mesh_from_latent_codes(c, style_a, self.classes[0]) for c in contents_ab]
            path_a = '/'.join([self.demo_pair_path, 'interpolations', 'swipe_content', '{}', self.classes[0]]) + \
                     f"_{self.classes[0]}.ply"
            for i, mesh_a in enumerate(out_a):
                self.save_mesh(mesh_a, path_a.format(i))
            # save meshes style_b
            print(f'Computing meshes for style {self.classes[1]}')
            out_b = [self.network.module.generate_mesh_from_latent_codes(c, style_b, self.classes[1]) for c in contents_ba]
            path_b = '/'.join([self.demo_pair_path, 'interpolations', 'swipe_content', '{}', self.classes[1]]) + \
                     f"_{self.classes[1]}.ply"
            for i, mesh_b in enumerate(out_b):
                self.save_mesh(mesh_b, path_b.format(i))


            if self.opt.share_decoder and self.opt.share_style_mlp:
                # Interpolate styles (here decoder class doesn't matter since it is shared)
                my_utils.magenta_print("Style interpolation...")
                styles_ab = np.linspace(style_a.cpu(), style_b.cpu(), self.num_interpolations)
                styles_ab = [torch.from_numpy(s).to(self.opt.device) for s in styles_ab]
                styles_ba = np.linspace(style_b.cpu(), style_a.cpu(), self.num_interpolations)
                styles_ba = [torch.from_numpy(s).to(self.opt.device) for s in styles_ba]
                out_a = [self.network.module.generate_mesh_from_latent_codes(content_a, s, self.classes[0])
                         for s in styles_ab]
                out_b = [self.network.module.generate_mesh_from_latent_codes(content_b, s, self.classes[1])
                         for s in styles_ba]
                path_a = '/'.join([self.demo_pair_path, 'interpolations', 'swipe_style', '{}']) + \
                         f"_{self.classes[0]}." + self.mesh_ext
                path_b = '/'.join([self.demo_pair_path, 'interpolations', 'swipe_style', '{}']) + \
                         f"_{self.classes[1]}." + self.mesh_ext
                for i, (mesh_a, mesh_b) in enumerate(zip(out_a, out_b)):
                    self.save_mesh(mesh_a, path_a.format(i), self.operation_a)
                    self.save_mesh(mesh_b, path_b.format(i), self.operation_b)
        else:
            print('No interpolation is possible when content and style encoders are not shared across domains.')

    def generate_k_mesh_interpolations(self, content_inputs, style_input_a, style_input_b):
        _, style_a = self.network.module.get_latent_codes_from_sample(style_input_a, self.classes[0])
        _, style_b = self.network.module.get_latent_codes_from_sample(style_input_b, self.classes[1])

        if self.opt.share_content_encoder:
            # You can interpolate across different styles and contents
            # Interpolate contents
            idx_a = 0
            idx_b = 0
            for i in range(len(content_inputs)):
                content_input_0 = content_inputs[i]
                content_input_1 = content_inputs[0] if i == (len(content_inputs)-1) else content_inputs[i+1]
                
                content_0, _ = self.network.module.get_latent_codes_from_sample(content_input_0, self.classes[0])
                content_1, _ = self.network.module.get_latent_codes_from_sample(content_input_1, self.classes[0])
                my_utils.magenta_print("Content interpolation...")
                contents_01 = np.linspace(content_0.cpu(), content_1.cpu(), self.num_interpolations)
                contents_01 = [torch.from_numpy(c).to(self.opt.device) for c in contents_01]
                # save meshes style_a
                print(f'Computing meshes for style {self.classes[0]}')
                out_a = [self.network.module.generate_mesh_from_latent_codes(c, style_a, self.classes[0]) for c in contents_01]
                path_a = os.path.join(self.demo_pair_path, 'interpolations', 'swipe_content', self.classes[0]) + \
                         "/{}_{}.ply"
                for j, mesh_a in enumerate(out_a):
                    idx_a += 1
                    self.save_mesh(mesh_a, path_a.format(idx_a, self.classes[0]))
                # save meshes style_b
                print(f'Computing meshes for style {self.classes[1]}')
                out_b = [self.network.module.generate_mesh_from_latent_codes(c, style_b, self.classes[1]) for c in contents_01]
                path_b = os.path.join(self.demo_pair_path, 'interpolations', 'swipe_content', self.classes[1]) + \
                         "/{}_{}.ply"
                for j, mesh_b in enumerate(out_b):
                    idx_b += 1
                    self.save_mesh(mesh_b, path_b.format(idx_b, self.classes[1]))

        else:
            print('No interpolation is possible when content and style encoders are not shared across domains.')

    def generate_mesh_with_style_noise(self, x, content_class, style_class, num_samples=10, operation=None,
                                       noise_magnitude=0.01, save=False):
        path = '/'.join([self.demo_pair_path, f"noise_{noise_magnitude}",
                         content_class + '_' + style_class, '{}.ply'])
        for i in range(num_samples):
            content, style = self.network.module.get_latent_codes_with_style_noise(x, content_class, style_class,
                                                                                   noise_magnitude)
            mesh = self.network.module.generate_mesh_from_latent_codes(content, style, style_class)
            if save:
                self.save_mesh(mesh, path.format(str(i).zfill(3)), operation)

    def generate_pairs_with_style_sampling(self, x, content_class, style_class, num_pairs=19):
        """Generate a pair of pointclouds given the same content code and two random style codes extracted from a
        gaussian distribution."""
        output_pairs = []
        content = self.network.module.get_latent_content(x, content_class)
        for i in range(num_pairs):
            style_a, style_b = self.network.module.get_pair_of_latent_styles_with_style_noise(
                x, content_class, style_class, self.opt.noise_magnitude)
            out_a = self.network.module.generate_pointcloud_from_latent_codes(content, style_a, style_class)
            out_b = self.network.module.generate_pointcloud_from_latent_codes(content, style_b, style_class)

            out_a = self.fuse_primitives(out_a['points_3'], None, False)
            out_b = self.fuse_primitives(out_b['points_3'], None, False)
            output_pairs.append((out_a, out_b))
        return output_pairs

    def compute_lpips_on_pairs(self, output_pairs):
        def make_network_input(points):
            return points.transpose(2, 1).contiguous()
        dist = 0.0
        num_pairs = 0
        for (out_a, out_b) in output_pairs:
            import time
            start = time.time()
            dist = self.perceptual_distance(make_network_input(out_a), make_network_input(out_b))
            print(time.time()-start)
            num_pairs += 1
        return dist/num_pairs

    def evaluate_average_lpips(self, num_inputs=100, num_pairs=19):
        dist_00, dist_01, dist_10, dist_11 = 0.0, 0.0, 0.0, 0.0
        for i in range(num_inputs):
            index_a = np.random.randint(self.datasets.len_dataset_test[self.classes[0]])
            data_a = EasyDict(self.datasets.dataset_test[self.classes[0]][index_a])
            index_b = np.random.randint(self.datasets.len_dataset_test[self.classes[1]])
            data_b = EasyDict(self.datasets.dataset_test[self.classes[1]][index_b])
            demo_path_a = data_a.pointcloud_path
            demo_path_b = data_b.pointcloud_path

            data_a = EasyDict(self.datasets.dataset_train[self.classes[0]].load(demo_path_a))
            data_b = EasyDict(self.datasets.dataset_train[self.classes[1]].load(demo_path_b))

            # prepare normalization
            self.make_network_input(data_a, self.SVR_0)
            self.make_network_input(data_b, self.SVR_1)
            x = {self.classes[0]: data_a.network_input,
                 self.classes[1]: data_b.network_input}
            pairs_00 = self.generate_pairs_with_style_sampling(x,
                                                               content_class=self.classes[0],
                                                               style_class=self.classes[0],
                                                               num_pairs=num_pairs)
            pairs_01 = self.generate_pairs_with_style_sampling(x,
                                                               content_class=self.classes[0],
                                                               style_class=self.classes[1],
                                                               num_pairs=num_pairs)
            pairs_10 = self.generate_pairs_with_style_sampling(x,
                                                               content_class=self.classes[1],
                                                               style_class=self.classes[0],
                                                               num_pairs=num_pairs)
            pairs_11 = self.generate_pairs_with_style_sampling(x,
                                                               content_class=self.classes[1],
                                                               style_class=self.classes[1],
                                                               num_pairs=num_pairs)
            dist_00 += self.compute_lpips_on_pairs(pairs_00)
            dist_01 += self.compute_lpips_on_pairs(pairs_01)
            dist_10 += self.compute_lpips_on_pairs(pairs_10)
            dist_11 += self.compute_lpips_on_pairs(pairs_11)

            print(
                '\r' + colored('[Input: %d/%d]' % (i, num_inputs), 'red') +'\n' +
                colored('lpips_00:  %f' % (dist_00/(i+1)).item(), 'yellow') +'\n' +
                colored('lpips_01:  %f' % (dist_01/(i+1)).item(), 'yellow') +'\n' +
                colored('lpips_10:  %f' % (dist_10/(i+1)).item(), 'yellow') +'\n' +
                colored('lpips_11:  %f' % (dist_11/(i+1)).item(), 'yellow') +'\n' +
                colored('average_lpips:  %f' % (torch.mean(torch.stack([dist_00, dist_01, dist_10, dist_11]))/(i+1)).item(), 'yellow'),
                end='')

        dist_00 /= num_inputs
        dist_01 /= num_inputs
        dist_10 /= num_inputs
        dist_11 /= num_inputs
        average = torch.mean(torch.stack([dist_00, dist_01, dist_10, dist_11]))
        return {'lpips_00': dist_00.item(),
                'lpips_01': dist_01.item(),
                'lpips_10': dist_10.item(),
                'lpips_11': dist_11.item(),
                'average_lpips': average.item()}

    def write_obj(self, verts, file_name):
        verts = verts.squeeze(0).cpu().numpy()
        with open(file_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

    def demo(self, demo_dir, classes):
        """
        This function takes a pointcloud path as input and save the mesh infered by Atlasnet
        Extension supported are ply npy obg and png
        :return: path to the generated mesh
        """
        my_utils.cyan_print("Running demo")

        generate_k_interpolations = False
        # generate_k_interpolations = True
        self.demo_pair_path = '/'.join([self.opt.demo_media_path, 'awesome'])
        if generate_k_interpolations:
            content_dir = '/home/mattia/Documents/3dsnet_video/interpolations/chairs/examples/contents'
            style_a_path = '/home/mattia/Documents/3dsnet_video/interpolations/chairs/examples/style_armchair.obj'
            style_b_path = '/home/mattia/Documents/3dsnet_video/interpolations/chairs/examples/style_chair.obj'
            content_inputs = []
            for f in os.listdir(content_dir):
                file_path = os.path.join(content_dir, f)
                content = EasyDict(self.datasets.dataset_train[self.classes[0]].load(file_path))
                self.make_network_input(content, self.SVR_0)
                content_inputs.append(content.network_input)

            style_input_a = EasyDict(self.datasets.dataset_train[self.classes[0]].load(style_a_path))
            style_input_b = EasyDict(self.datasets.dataset_train[self.classes[1]].load(style_b_path))
            self.make_network_input(style_input_a, self.SVR_0)
            self.make_network_input(style_input_b, self.SVR_1)
            style_input_a = style_input_a.network_input
            style_input_b = style_input_b.network_input
            self.generate_k_mesh_interpolations(content_inputs, style_input_a, style_input_b)

        for pair_index in range(self.opt.num_demo_pairs):
            my_utils.red_print(f"Pair #{pair_index}")
            self.demo_pair_path = '/'.join([self.opt.demo_media_path, str(pair_index)])

            if not os.path.exists(self.demo_pair_path):
                os.mkdir(self.demo_pair_path)

            if self.opt.use_default_demo_samples:
                if self.opt.dataset == 'SMXL':
                    demo_path_a = '/'.join([demo_dir, classes[0]]) + '.obj'
                    demo_path_b = '/'.join([demo_dir, classes[1]]) + '.obj'
                elif self.opt.dataset == 'ShapeNet':
                    demo_path_a = '/'.join([demo_dir, classes[0]]) + '.points.ply.npy'
                    demo_path_b = '/'.join([demo_dir, classes[1]]) + '.points.ply.npy'
            else:
                index_a = np.random.randint(self.datasets.len_dataset_test[self.classes[0]])
                data_a = EasyDict(self.datasets.dataset_test[self.classes[0]][index_a])
                index_b = np.random.randint(self.datasets.len_dataset_test[self.classes[1]])
                data_b = EasyDict(self.datasets.dataset_test[self.classes[1]][index_b])
                demo_path_a = data_a.pointcloud_path
                demo_path_b = data_b.pointcloud_path

            data_a = EasyDict(self.datasets.dataset_train[self.classes[0]].load(demo_path_a))
            data_b = EasyDict(self.datasets.dataset_train[self.classes[1]].load(demo_path_b))

            # prepare normalization
            self.make_network_input(data_a, self.SVR_0)
            self.make_network_input(data_b, self.SVR_1)
            x = {self.classes[0]: data_a.network_input,
                 self.classes[1]: data_b.network_input}
            self.set_operation(data_a, data_b)
            if not self.opt.use_default_demo_samples:
                if os.path.exists(demo_path_a) and os.path.exists(demo_path_b):
                    copy_path_a = '/'.join([self.demo_pair_path, str(self.flags.media_count)]) + \
                                  f"_input_{self.classes[0]}"
                    copy_path_b = '/'.join([self.demo_pair_path, str(self.flags.media_count)]) + \
                                  f"_input_{self.classes[1]}"
                    shutil.copyfile(demo_path_a, copy_path_a + '.npy')
                    shutil.copyfile(demo_path_b, copy_path_b + '.npy')
                    self.write_obj(data_a.points, copy_path_a + '_points.obj')
                    self.write_obj(data_b.points, copy_path_b + '_points.obj')
                path_a, path_b = self.copy_input_meshes(data_a, data_b, self.demo_pair_path)
                path_a, path_b = self.copy_input_renderings(data_a, data_b, self.demo_pair_path)

            # Get results of forward pass
            my_utils.yellow_print("Generating basic style transfer results...")
            path_aa, _ = self.generate_mesh_from_classes(x, self.classes[0], self.classes[0], self.operation_a, demo=True)
            path_ab, _ = self.generate_mesh_from_classes(x, self.classes[0], self.classes[1], self.operation_a, demo=True)
            path_bb, _ = self.generate_mesh_from_classes(x, self.classes[1], self.classes[1], self.operation_b, demo=True)
            path_ba, _ = self.generate_mesh_from_classes(x, self.classes[1], self.classes[0], self.operation_b, demo=True)

            # Interpolate across latent spaces
            my_utils.yellow_print("Interpolating latent codes...")
            self.generate_mesh_interpolations(x)

            # Generate samples with sampled styles
            my_utils.yellow_print("Generating meshes with additive noise on style latent codes...")
            self.generate_mesh_with_style_noise(x, self.classes[0], self.classes[0], 10, self.operation_a,
                                                self.opt.noise_magnitude, save=True)
            self.generate_mesh_with_style_noise(x, self.classes[0], self.classes[1], 10, self.operation_a,
                                                self.opt.noise_magnitude, save=True)
            self.generate_mesh_with_style_noise(x, self.classes[1], self.classes[1], 10, self.operation_b,
                                                self.opt.noise_magnitude, save=True)
            self.generate_mesh_with_style_noise(x, self.classes[1], self.classes[0], 10, self.operation_b,
                                                self.opt.noise_magnitude, save=True)

            if self.opt.use_default_demo_samples:
                break

    def set_operation(self, data_a, data_b):
        # Get normalization ops
        self.operation_a, self.operation_b = None, None
        # Unnormalize like domain a
        if hasattr(data_a, 'operation') and data_a.operation is not None:
            data_a.operation.invert()
            self.operation_a = data_a.operation
        # Unnormalize like domain b
        if hasattr(data_b, 'operation') and data_b.operation is not None:
            data_b.operation.invert()
            self.operation_b = data_b.operation

def unnormalize(mesh, operation=None):
    if operation is not None:
        # Undo any normalization that was used to preprocess the input.
        vertices = torch.from_numpy(mesh.vertices).clone().unsqueeze(0)
        unnormalized_vertices = operation.apply(vertices)
        mesh = pymesh.form_mesh(vertices=unnormalized_vertices.squeeze().numpy(), faces=mesh.faces)
        return mesh

def rename_path(path, unnormalized=False, demo=False, interpolated=False, ext='ply'):
    path = path.split('.')
    if unnormalized:
        path[-2] += "_unnormalized"
    path[-1] = ext
    path = ".".join(path)
    if demo:
        path = path.split('/')
        path[-3] = "demo_media"
        if interpolated:
            path[-2] += "/interpolated"
        demo_media_path = "/".join(path[:-1])
        if not os.path.exists(demo_media_path):
            os.mkdir(demo_media_path)
        path = "/".join(path)
    return path

def pointcloud_to_shapenet_path(path):
    path = path.split('/')
    path[-3] = "ShapeNetV1Core"
    file_name = path[-1].split('.')
    path[-1] = file_name[0]
    path = "/".join(path)
    path = os.path.join(path, 'model.obj')
    return path

def pointcloud_to_renderings_path(path):
    path = path.split('/')
    path[-3] = "ShapeNetV1Renderings"
    file_name = path[-1].split('.')
    path[-1] = file_name[0]
    path = "/".join(path)
    path = os.path.join(path, 'rendering')
    return path

