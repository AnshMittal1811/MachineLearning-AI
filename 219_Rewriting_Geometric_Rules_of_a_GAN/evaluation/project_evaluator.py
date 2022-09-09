import os
import numpy as np
from PIL import Image

import util
from evaluation import BaseEvaluator
from lib import quality_assess


class ProjectEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def evaluate(self, model, dataset, nsteps=None):
        assert type(model) == tuple, "need a pair of (reference, updated) model to apply this visualization."

        # set up the visualization webpage
        if nsteps is None:
            savedir = os.path.join(self.output_dir(), "%s" % self.target_phase)
            webpage_title = "Selected samples of %s. phase=%s" % \
                            (self.opt.name, self.target_phase)
        else:
            savedir = os.path.join(self.output_dir(), "%s_%s" % (self.target_phase, nsteps))
            webpage_title = "Selected samples of %s. iter=%s. phase=%s" % \
                            (self.opt.name, str(nsteps), self.target_phase)
        os.makedirs(savedir, exist_ok=True)
        webpage = util.HTML(savedir, webpage_title)

        # some bookkeeping
        ref_model, new_model = model

        device = ref_model.device
        ref_G = ref_model.netG
        new_G = new_model.netG

        # setting up evaluation and visualization
        lpips_metric = quality_assess.LPIPSMetric(device)
        metrics_keys = ['mse', 'psnr', 'ssim', 'lpips']
        metrics = {f'{self.target_phase}_{k}': [] for k in metrics_keys}
        visuals_keys = ['im_orig', 'im_edited', 'im_target']
        visuals = {f'{self.target_phase}_{k}': [] for k in visuals_keys}

        # performs evaluation and visualization
        dset = dataset.dataloader.dataset
        for i in range(len(dset)):
            # process data
            latents = dset.data['latents'][i].unsqueeze(0).to(device)

            # get image from the original model
            orig = ref_G.synthesis(latents, noise_mode='const')
            orig_np = util.tensor2im(orig[0], tile=False, normalize=True)
            orig_im = Image.fromarray(orig_np)

            # get image from the updated model
            edited = new_G.synthesis(latents, noise_mode='const')
            edited_np = util.tensor2im(edited[0], tile=False, normalize=True)
            edited_im = Image.fromarray(edited_np)

            # get warped target
            target = dset.data['targets'][i]
            target_np = util.tensor2im(target, tile=False, normalize=True)
            target_im = Image.fromarray(target_np)

            # add stuff to the webpage
            images = [orig_im, target_im, edited_im]
            texts = ["%s_%04d.png" % (label, i) for label in ['orig', 'target', 'edited']]
            webpage.add_images(images, texts, height=400)

            # gather visuals
            visuals[f'{self.target_phase}_im_orig'].append(orig_np)
            visuals[f'{self.target_phase}_im_edited'].append(edited_np)
            visuals[f'{self.target_phase}_im_target'].append(target_np)

            # collect errors
            edited_eval = edited_np.astype(np.float64)
            target_eval = target_np.astype(np.float64)
            mse = quality_assess.mse(edited_eval, target_eval)
            psnr = quality_assess.psnr(edited_eval, target_eval)
            ssim = quality_assess.ssim(edited_eval, target_eval)
            lpips = lpips_metric.eval_single_pair(edited_eval, target_eval)
            metrics[f'{self.target_phase}_mse'].append(mse)
            metrics[f'{self.target_phase}_psnr'].append(psnr)
            metrics[f'{self.target_phase}_ssim'].append(ssim)
            metrics[f'{self.target_phase}_lpips'].append(lpips)

        # update metrics/visuals and save the webpage
        lpips_metric.close()
        metrics = {k: np.mean(v) for k, v in metrics.items()}
        visuals = {k: np.hstack(v) for k, v in visuals.items()}
        webpage.save()

        return metrics, visuals
