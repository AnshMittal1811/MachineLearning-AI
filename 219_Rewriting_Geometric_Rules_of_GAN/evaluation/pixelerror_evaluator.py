import os
import numpy as np
from PIL import Image

import util
from lib import quality_assess
from evaluation import BaseEvaluator


class PixelErrorEvaluator(BaseEvaluator):
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
        metrics_keys = ['psnr', 'psnr_changed', 'psnr_unchanged', 'ssim', 'ssim_changed', 'ssim_unchanged', 'lpips', 'lpips_changed', 'lpips_unchanged']
        metrics = {f'{self.target_phase}_{k}': [] for k in metrics_keys}
        visuals_keys = ['im_orig', 'im_edited', 'im_target', 'im_overlay', 'err_l1', 'err_l1_changed', 'err_l1_unchanged', 'err_lpips']
        visuals = {f'{self.target_phase}_{k}': [] for k in visuals_keys}

        # performs evaluation and visualization
        lpips_metric = quality_assess.LPIPSMetric(device)
        lpips_spatial = quality_assess.LPIPSMetric(device, spatial=True)
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

            # get target mask
            target_mask = dset.data['target_masks'][i]
            changed_mask = target_mask.cpu().numpy()[..., None]
            unchanged_mask = 1 - changed_mask
            overlay_np = ((changed_mask * 0.5 + 0.5) * edited_np).astype(np.uint8)
            overlay_im = Image.fromarray(overlay_np)

            # collect errors
            edited_eval = edited_np.astype(np.float64)
            target_eval = target_np.astype(np.float64)
            psnr = quality_assess.psnr(edited_eval, target_eval)
            ssim = quality_assess.ssim(edited_eval, target_eval)
            lpips = lpips_metric.eval_single_pair(edited_eval, target_eval)
            metrics[f'{self.target_phase}_psnr'].append(psnr)
            metrics[f'{self.target_phase}_ssim'].append(ssim)
            metrics[f'{self.target_phase}_lpips'].append(lpips)

            # masked stuff
            psnr_changed = quality_assess.psnr(edited_eval, target_eval, mask=changed_mask)
            ssim_changed = quality_assess.ssim(edited_eval, target_eval, mask=changed_mask)
            lpips_changed = lpips_spatial.eval_single_pair_masked(edited_eval, target_eval, changed_mask[..., 0])

            psnr_unchanged = quality_assess.psnr(edited_eval, target_eval, mask=unchanged_mask)
            ssim_unchanged = quality_assess.ssim(edited_eval, target_eval, mask=unchanged_mask)
            lpips_unchanged = lpips_spatial.eval_single_pair_masked(edited_eval, target_eval, unchanged_mask[..., 0])

            metrics[f'{self.target_phase}_psnr_changed'].append(psnr_changed)
            metrics[f'{self.target_phase}_ssim_changed'].append(ssim_changed)
            metrics[f'{self.target_phase}_lpips_changed'].append(lpips_changed)

            metrics[f'{self.target_phase}_psnr_unchanged'].append(psnr_unchanged)
            metrics[f'{self.target_phase}_ssim_unchanged'].append(ssim_unchanged)
            metrics[f'{self.target_phase}_lpips_unchanged'].append(lpips_unchanged)

            # visualize errors
            l1_errormap = np.linalg.norm(edited_eval - target_eval, ord=1, axis=2)
            l1_errormap_pt = util.plot_spatialmap(l1_errormap)
            l1_emap_changed = util.plot_spatialmap(changed_mask[..., 0] * l1_errormap)
            l1_emap_unchanged = util.plot_spatialmap(unchanged_mask[..., 0] * l1_errormap)

            lpips_errormap = lpips_spatial.eval_single_pair(edited_eval, target_eval)
            lpips_errormap = util.plot_spatialmap(lpips_errormap)

            # add stuff to the webpage
            images = [orig_im, target_im, edited_im, overlay_im, l1_errormap_pt, l1_emap_changed, l1_emap_unchanged, lpips_errormap]
            texts = ['original', 'target', 'edited', 'edited_overlay', f'PSNR: {psnr}', f'PSNR_changed: {psnr_changed}', f'PSNR_unchanged: {psnr_unchanged}', 'lpips']
            links = ["%s_%04d.png" % (label, i) for label in ['orig', 'target', 'edited', 'overlay', 'l1', 'l1_changed', 'l1_unchanged', 'lpips']]
            webpage.add_images(images, texts, links=links, height=400)

            # gather visuals
            emaps_np = [np.asarray(emap).copy() for emap in [l1_errormap_pt, l1_emap_changed, l1_emap_unchanged, lpips_errormap]]
            images = [orig_np, edited_np, target_np, overlay_np] + emaps_np
            for vis, vis_key in zip(images, visuals_keys):
                visuals[f'{self.target_phase}_{vis_key}'].append(vis)

        # update metrics/visuals and save the webpage
        lpips_metric.close()
        lpips_spatial.close()
        metrics = {k: np.mean(v) for k, v in metrics.items()}
        visuals = {k: np.hstack(v) for k, v in visuals.items()}
        webpage.save()

        return metrics, visuals
