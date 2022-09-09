import os
from PIL import Image
import numpy as np
import torch
from evaluation import BaseEvaluator
import util


class SampleEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--random_sample_seed", type=int, default=3000, help="seed to generate random samples.")
        parser.add_argument("--num_random_images", type=int, default=200, help="number of random samples.")
        parser.add_argument("--random_sample_batch", type=int, default=10, help="batch size to generate random samples.")
        parser.add_argument("--random_sample_trunc", type=float, default=1.0, help="truncation applied on random samples.")
        return parser

    def evaluate(self, model, dataset, nsteps=None):
        assert type(model) == tuple, "need a pair of (reference, updated) model to apply this visualization."

        # set up the visualization webpage
        if nsteps is None:
            savedir = os.path.join(self.output_dir(), "%s_%s" % (self.target_phase, nsteps))
            webpage_title = "Random sample of %s. iter=%s. phase=%s" % \
                            (self.opt.name, str(nsteps), self.target_phase)
        else:
            savedir = os.path.join(self.output_dir(), "%s" % self.target_phase)
            webpage_title = "Random sample of %s. phase=%s" % \
                            (self.opt.name, self.target_phase)
        os.makedirs(savedir, exist_ok=True)
        webpage = util.HTML(savedir, webpage_title)

        # some bookkeeping
        ref_model, new_model = model

        N = self.opt.num_random_images
        B = self.opt.random_sample_batch
        seed = self.opt.random_sample_seed
        trunc = self.opt.random_sample_trunc
        device = ref_model.device

        ref_G = ref_model.netG
        new_G = new_model.netG
        z_dim = ref_G.z_dim

        # generate random samples and save to the webpage
        z = torch.from_numpy(np.random.RandomState(seed).randn(N, z_dim)).to(device)

        # samples from the original model
        images_orig = []
        for start in range(0, N, B):
            end = min(N, start + B)
            images_orig.append(ref_G(z[start:end], None, truncation_psi=trunc))
        images_orig = torch.cat(images_orig, dim=0)
        images_orig = util.tensor2im(images_orig, tile=False, normalize=True)
        vis_orig = np.hstack(images_orig[:8])
        # convert to PIL.Image and save to webpage
        images_orig = [Image.fromarray(im) for im in images_orig]
        webpage.add_images(images_orig, ["orig_%04d.png" % i for i in range(N)], height=400)
        del images_orig

        # samples from the edited model
        images_mod = []
        for start in range(0, N, B):
            end = min(N, start + B)
            images_mod.append(new_G(z[start:end], None, truncation_psi=trunc))
        images_mod = torch.cat(images_mod, dim=0)
        images_mod = util.tensor2im(images_mod, tile=False, normalize=True)
        vis_mod = np.hstack(images_mod[:8])
        # convert to PIL.Image and save to webpage
        images_mod = [Image.fromarray(im) for im in images_mod]
        webpage.add_images(images_mod, ["edited_%04d.png" % i for i in range(N)], height=400)

        # generate visualization and save the webpage
        visual = np.vstack([vis_orig, vis_mod])
        webpage.save()
        return {}, {f'{self.target_phase}_sample': visual}
