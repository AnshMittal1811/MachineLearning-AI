import os
import glob
import wandb
import ntpath
import time
import numpy as np

from . import util
from . import html


class Visualizer():
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--wandb_project", default='GAN Warping')
        parser.add_argument("--wandb_tags", default='misc')
        parser.add_argument("--no_html", action='store_true')
        parser.add_argument("--no_wandb", action='store_true')
        parser.add_argument("--display_winsize", default=400, help='display window size')

        return parser

    def __init__(self, opt):
        self.opt = opt
        self.wandb_log = opt.isTrain and not opt.no_wandb
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name

        # setup wandb
        if self.wandb_log:
            if opt.resume_iter is None:
                wandb_tags = opt.wandb_tags.split(',')
                wandb.init(project=opt.wandb_project, name=self.name, tags=wandb_tags)
                wandb.config.update(opt)
                # save wandb run id to a file
                save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "wandb_id.txt")
                with open(save_path, "w") as f:
                    f.write(wandb.run.id + '\n')
            else:
                # when resume job, also resume the wandb run by the saved id
                load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "wandb_id.txt")
                with open(load_path, "r") as f:
                    wandb_id = f.readline().strip()
                wandb.init(resume=wandb_id)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            self.iter_log = []
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, step, visuals, disable_html=False):

        # convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)

        if self.wandb_log:  # show images in tensorboard output
            img_summaries = {}
            for label, image_numpy in visuals.items():
                assert len(image_numpy.shape) == 3, "visualization should be be a single RGB image"
                # Create an Image object
                image_numpy = util.clip_image_size(image_numpy, max_size=1024)
                img_summaries[label] = [wandb.Image(image_numpy, caption=label)]

            # Write Summary
            wandb.log(img_summaries, step=step)

        if self.use_html and not disable_html:  # save images to a html file
            for label, image_numpy in visuals.items():
                assert len(image_numpy.shape) == 3, "visualization should be be a single RGB image"
                img_path = os.path.join(self.img_dir, 'iter%.3d_%s.png' % (step, label))
                image_numpy = util.clip_image_size(image_numpy, max_size=1024)
                util.save_image(image_numpy, img_path)

            # update website
            self.iter_log.append(step)
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)
            for itr in self.iter_log[::-1]:
                webpage.add_header('iter [%d]' % itr)
                ims = []
                txts = []
                links = []

                imfiles = sorted(glob.glob(os.path.join(self.img_dir, 'iter%.3d*' % itr)))
                for fname in imfiles:
                    base = os.path.basename(fname)
                    info = os.path.splitext(base)[0].split('_')
                    # note this contains substring epoch, iter
                    step = info[0]
                    label = '_'.join(info[1:])
                    txt = label
                    # make sure only include visuals wrt to the exact iteration
                    if step.replace('iter', '') == f'{itr:03d}':
                        ims.append(base)
                        txts.append(txt)
                        links.append(base)

                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, height=self.win_size)
                else:
                    num = int(round(len(ims) / 2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], height=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], height=self.win_size)

            webpage.save()

    # logs: dictionary of error labels and values
    def plot_current_logs(self, iters, logs):
        if self.wandb_log:
            logs_mean = {k: v.mean().item() for k, v in logs.items()}
            wandb.log(logs_mean, step=iters)

    # summaries: dictionary of error labels and values
    def plot_current_summaries(self, summaries):
        if self.wandb_log:
            for k, v in summaries.items():
                wandb.summary[k] = v

    # losses: same format as |losses| of plot_current_losses
    def print_current_logs(self, iters, times, logs):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losgs (OrderedDict) -- training losses or metrics stored in the format of (name, float) pairs
            times (OrderedDict) -- running time per data point for processes in the format of (name, float)
        """
        message = '(iters: %d' % (iters)
        for k, v in times.items():
            message += ", %s: %.3f" % (k, v)
        message += ") "
        for k, v in logs.items():
            message += '%s: %.3f ' % (k, v.mean())

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            if isinstance(t, np.ndarray):
                continue
            tile = min(8, t.size(0)) if t.dim() == 4 else False
            t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        visuals = self.convert_visuals_to_numpy(visuals)

        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
