from termcolor import colored
from easydict import EasyDict
import os

class TrainerIteration(object):
    """
        This class implements all functions related to a single forward pass of 3DSNet.
    """

    def __init__(self, opt):
        super(TrainerIteration, self).__init__(opt)

    def make_network_input(self, data, svr=False):
        """
        Arrange to data to be fed to the network.


        :param data: data dictionary for the current batch of a given domain
        :param svr: True if the chosen input domain is image
        """
        data.network_input = {}
        data.network_input['points'] = data.points.transpose(2, 1).contiguous().to(self.opt.device)
        if svr:
            data.network_input['image'] = data.image.to(self.opt.device)
            data.network_input['svr'] = True
        else:
            data.network_input['svr'] = False

    def handle_outputs(self, data, out):
        # Merge generated surface elements in a single one and prepare data for Chamfer
        if 'reconstruction' in out and out['reconstruction'] is not None:
            data.reconstruction = self.fuse_primitives(out.reconstruction, out.faces, self.opt.multiscale_loss)
        if 'reconstruction_1' in out and out['reconstruction_1'] is not None:
            data.reconstruction_1 = self.fuse_primitives(out.reconstruction_1, out.faces, False)
        if 'reconstruction_2' in out and out['reconstruction_2'] is not None:
            data.reconstruction_2 = self.fuse_primitives(out.reconstruction_2, out.faces, self.opt.multiscale_loss)
        # Merge generated surface elements in a single one and prepare data for visualization
        if 'style_transfer' in out:
            data.style_transfer = self.fuse_primitives(out.style_transfer, out.faces, self.opt.multiscale_loss)
        # Merge generated surface elements in a single one and prepare data for cycle-consistent Chamfer
        if self.cycle_reconstruction and 'cycle_reconstruction' in out and out['cycle_reconstruction'] is not None:
            data.cycle_reconstruction = self.fuse_primitives(out.cycle_reconstruction,
                                                             out.faces,
                                                             self.opt.multiscale_loss)
        if self.cycle_reconstruction and 'cycle_reconstruction_1' in out  and out['cycle_reconstruction_1'] is not None:
            data.cycle_reconstruction_1 = self.fuse_primitives(out.cycle_reconstruction_1,
                                                               out.faces,
                                                               False)
        if self.cycle_reconstruction and 'cycle_reconstruction_2' in out  and out['cycle_reconstruction_2'] is not None:
            data.cycle_reconstruction_2 = self.fuse_primitives(out.cycle_reconstruction_2,
                                                               out.faces,
                                                               self.opt.multiscale_loss)
        if 'reconstruction_logits' in out:
            data.reconstruction_logits = out.reconstruction_logits
        if 'style_transfer_logits' in out:
            data.style_transfer_logits = out.style_transfer_logits
        if 'content_code' in out:
            data.content_code = out.content_code
        if 'style_code' in out:
            data.style_code = out.style_code
        if 'cycle_content_code' in out:
            data.cycle_content_code = out.cycle_content_code
        if 'cycle_style_code' in out:
            data.cycle_style_code = out.cycle_style_code

    def discriminator_ops(self, data_a, data_b):
        """
        Commom operations between train and eval forward passes during discriminator update

        :param data_a: data dictionary for the current batch of the first domain
        :param data_b: data dictionary for the current batch of the second domain
        """
        self.make_network_input(data_a, self.SVR_0)
        self.make_network_input(data_b, self.SVR_1)
        self.batch_size = data_a.points.size(0)
        # Feed input dictionary to training_forward pass
        x = {self.classes[0]: data_a.network_input,
             self.classes[1]: data_b.network_input}

        out_0, out_1 = self.network.module.discriminator_update_forward(x, train=self.flags.train)
        out_0, out_1 = EasyDict(out_0), EasyDict(out_1)

        self.handle_outputs(data_a, out_0)
        self.handle_outputs(data_b, out_1)
        self.discriminator_loss_model(data_a, data_b)  # batch

        if self.opt.use_visdom:
            self.visualize(data_a, self.classes[0], self.SVR_0)
            self.visualize(data_b, self.classes[1], self.SVR_1)

    def generator_ops(self, data_a, data_b):
        """
        Commom operations between train and eval forward passes during generator update

        :param data_a: data dictionary for the current batch of the first domain
        :param data_b: data dictionary for the current batch of the second domain
        """
        self.make_network_input(data_a, self.SVR_0)
        self.make_network_input(data_b, self.SVR_1)
        self.batch_size = data_a.points.size(0)
        # plt.imshow(data_a.network_input[0].permute(1,2,0).cpu())
        # Feed input dictionary to training_forward pass
        x = {self.classes[0]: data_a.network_input,
             self.classes[1]: data_b.network_input}

        out_0, out_1 = self.network.module.generator_update_forward(x, train=self.flags.train)
        out_0, out_1 = EasyDict(out_0), EasyDict(out_1)

        self.handle_outputs(data_a, out_0)
        self.handle_outputs(data_b, out_1)
        self.generator_loss_model(data_a, data_b)  # batch

        if self.opt.use_visdom:
            self.visualize(data_a, self.classes[0], self.SVR_0)
            self.visualize(data_b, self.classes[1], self.SVR_1)

    def discriminator_iteration(self, data_a, data_b):
        """
        One training iteration during discriminator update

        :param data_a: data dictionary for the current batch of the first domain
        :param data_b: data dictionary for the current batch of the second domain
        """
        data_a.loss, data_b.loss = 0, 0
        data_a.lpips_from_target, data_b.lpips_from_target = 0, 0
        data_a.lpips_from_source, data_b.lpips_from_source = 0, 0
        data_a.lpips_rec_from_source, data_b.lpips_rec_from_source = 0, 0
        self.discriminator_optimizer.zero_grad()
        self.discriminator_ops(data_a, data_b)
        loss = data_a.loss + data_b.loss
        self.log.update("loss_train_dis", loss.item())
        if not self.opt.no_learning:
            loss.backward()
            self.discriminator_optimizer.step()  # gradient update
        self.print_iteration_stats(loss, 'discriminator')

    def generator_iteration(self, data_a, data_b):
        """
        One training iteration during generator update

        :param data_a: data dictionary for the current batch of the first domain
        :param data_b: data dictionary for the current batch of the second domain
        """
        data_a.loss, data_b.loss = 0, 0
        data_a.lpips_from_target, data_b.lpips_from_target = 0, 0
        data_a.lpips_from_source, data_b.lpips_from_source = 0, 0
        data_a.lpips_rec_from_source, data_b.lpips_rec_from_source = 0, 0
        self.generator_optimizer.zero_grad()
        self.generator_ops(data_a, data_b)
        loss = data_a.loss + data_b.loss
        self.log.update("loss_train_gen", loss.item())
        if not self.opt.no_learning:
            loss.backward()
            self.generator_optimizer.step()  # gradient update
        self.print_iteration_stats(loss, 'generator')

    def train_iteration(self, data_a, data_b):
        """
        One training iteration

        :param data_a: data dictionary for the current batch of the first domain
        :param data_b: data dictionary for the current batch of the second domain
        """
        if self.iteration % self.opt.generator_update_skips == 0:
            self.generator_iteration(data_a, data_b)
        if self.iteration % self.opt.discriminator_update_skips == 0:
            self.discriminator_iteration(data_a, data_b)

    def visualize(self, data, category, svr=False):
        if self.iteration % 50 == 1:
            tmp_string = "train" if self.flags.train else "test"
            self.visualizer.show_pointcloud(data.points[0], title=f"GT {tmp_string} {category}")
            self.visualizer.show_pointcloud(data.reconstruction[0], title=f"Reconstruction {tmp_string} {category}")
            self.visualizer.show_pointcloud(data.style_transfer[0], title=f"It was a {tmp_string} {category}")
            if svr:
                self.visualizer.show_image(data.image[0], title=f"Input Image {tmp_string}")

    def test_iteration(self, data_a, data_b):
        """
        One test iteration

        :param data_a: data dictionary for the current batch of the first domain
        :param data_b: data dictionary for the current batch of the second domain
        """
        data_a.loss, data_b.loss = 0, 0
        data_a.lpips_from_target, data_b.lpips_from_target = 0, 0
        data_a.lpips_from_source, data_b.lpips_from_source = 0, 0
        data_a.lpips_rec_from_source, data_b.lpips_rec_from_source = 0, 0
        data_a.lpips_rec_from_target, data_b.lpips_rec_from_target = 0, 0
        self.generator_ops(data_a, data_b)
        # total loss
        loss = data_a.loss + data_b.loss
        # lpips
        lpips_from_target = data_a.lpips_from_target + data_b.lpips_from_target
        lpips_from_source = data_a.lpips_from_source + data_b.lpips_from_source
        lpips_rec_from_source = data_a.lpips_rec_from_source + data_b.lpips_rec_from_source
        lpips_rec_from_target = data_a.lpips_rec_from_target + data_b.lpips_rec_from_target
        # Style Transfer Score (STS) computation
        delta_source_a = data_a.lpips_from_source.abs()-data_a.lpips_rec_from_source.abs()
        delta_target_a = data_a.lpips_from_target.abs()-data_a.lpips_rec_from_target.abs()
        delta_source_b = data_b.lpips_from_source.abs()-data_b.lpips_rec_from_source.abs()
        delta_target_b = data_b.lpips_from_target.abs()-data_b.lpips_rec_from_target.abs()
        style_transfer_score_a = delta_source_a - delta_target_a
        style_transfer_score_b = delta_source_b - delta_target_b
        # Overall Score
        delta_source = delta_source_a + delta_source_b
        delta_target = delta_target_a + delta_target_b
        style_transfer_score = style_transfer_score_a + style_transfer_score_b

        loss_fscore = data_a.loss_fscore + data_b.loss_fscore
        chamfer_distance = data_a.chamfer_distance + data_b.chamfer_distance
        self.num_val_points = data_a.reconstruction.size(1)
        self.log.update("loss_val", loss.item())
        if os.path.exists(self.opt.reload_pointnet_path):
            self.log.update("lpips_rec_from_source", lpips_rec_from_source.item())
            self.log.update("lpips_rec_from_target", lpips_rec_from_target.item())
            self.log.update("lpips_from_source", lpips_from_source.item())
            self.log.update("lpips_from_target", lpips_from_target.item())
            self.log.update("delta_source", delta_source.item())
            self.log.update("delta_target", delta_target.item())
            self.log.update("style_transfer_score", style_transfer_score.item())
        self.log.update("fscore", loss_fscore.item())
        self.log.update("chamfer_distance", chamfer_distance.item())

        self.last_fscore = loss_fscore.item()

        len_dataset_test = self.datasets.min_len_dataset_test
        print(
            '\r' + colored(
                '[%d: %d/%d]' % (self.epoch, self.iteration, len_dataset_test / self.opt.batch_size_test),
                'red') +
            colored('loss_val:  %f' % (data_a.loss + data_b.loss).item(), 'yellow'),
            end='')
