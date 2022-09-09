import os
import cv2
import torch
import numpy as np
from PIL import Image
from skimage.morphology import remove_small_objects

import util
from evaluation.dexined_net import get_dexined_net
from evaluation import BaseEvaluator


class ChamferEvaluator(BaseEvaluator):
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
        metrics_keys = ['chamfer', 'chamfer_changed', 'chamfer_unchanged']
        metrics = {f'{self.target_phase}_{k}': [] for k in metrics_keys}
        visuals_keys = ['edge_edited', 'edge_target']
        visuals = {f'{self.target_phase}_{k}': [] for k in visuals_keys}

        # performs evaluation and visualization
        edge_net = get_dexined_net().to(device)
        edge_net.eval()
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
            changed_mask = target_mask.cpu().numpy().astype(bool)
            unchanged_mask = ~changed_mask

            # get edge predictions
            edited_edge = get_edge(edge_net, edited_np, device)
            target_edge = get_edge(edge_net, target_np, device)
            edited_edge_im = Image.fromarray(edited_edge)
            target_edge_im = Image.fromarray(target_edge)

            edited_edge_changed = edited_edge.copy()
            edited_edge_changed[~changed_mask] = 255
            target_edge_changed = target_edge.copy()
            target_edge_changed[~changed_mask] = 255

            edited_edge_unchanged = edited_edge.copy()
            edited_edge_unchanged[~unchanged_mask] = 255
            target_edge_unchanged = target_edge.copy()
            target_edge_unchanged[~unchanged_mask] = 255

            edited_edge_changed_im = Image.fromarray(edited_edge_changed)
            target_edge_changed_im = Image.fromarray(target_edge_changed)
            edited_edge_unchanged_im = Image.fromarray(edited_edge_unchanged)
            target_edge_unchanged_im = Image.fromarray(target_edge_unchanged)

            # collect errors
            chamfer = symmetric_chamfer(edited_edge, target_edge)
            chamfer_changed = symmetric_chamfer(edited_edge_changed, target_edge_changed)
            chamfer_unchanged = symmetric_chamfer(edited_edge_unchanged, target_edge_unchanged)
            metrics[f'{self.target_phase}_chamfer'] = chamfer
            metrics[f'{self.target_phase}_chamfer_changed'] = chamfer_changed
            metrics[f'{self.target_phase}_chamfer_unchanged'] = chamfer_unchanged

            # add stuff to the webpage
            images = [orig_im, target_im, edited_im, target_edge_im, edited_edge_im, target_edge_changed_im, edited_edge_changed_im, target_edge_unchanged_im, edited_edge_unchanged_im]
            texts = [f'CD: {chamfer}\nCD(changed): {chamfer_changed}\nCD(unchanged): {chamfer_unchanged}',
                     'target', 'edited', 'target_edge', 'edited_edge', 'tgt_edge_changed', 'edt_edge_changed', 'tgt_edge_unchanged', 'edt_edge_unchanged']
            links = ["%s_%04d.png" % (label, i) for label in ['orig', 'target', 'edited', 'target_edge', 'edited_edge', 'tgt_edge_changed', 'edt_edge_changed', 'tgt_edge_unchanged', 'edt_edge_unchanged']]
            webpage.add_images(images, texts, links=links, height=400)

            # gather visuals
            images = [edited_edge[..., None].repeat(3, axis=-1), target_edge[..., None].repeat(3, axis=-1)]
            for vis, vis_key in zip(images, visuals_keys):
                visuals[f'{self.target_phase}_{vis_key}'].append(vis)

        # update metrics/visuals and save the webpage
        edge_net.to('cpu')
        del edge_net
        metrics = {k: np.mean(v) for k, v in metrics.items()}
        visuals = {k: np.hstack(v) for k, v in visuals.items()}
        webpage.save()

        return metrics, visuals


def edge_predict(edge_net, image, device):
    mean_bgr = np.array([103.939, 116.779, 123.68])
    data = np.array(image, np.float32)[..., ::-1].copy()
    data -= mean_bgr
    data = data.transpose((2, 0, 1))
    data = torch.from_numpy(data).float().unsqueeze(0).to(device)
    out = edge_net(data)
    out = torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
    return out


def non_maximum_suppression(image, angles):
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])

            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed


def double_threshold_hysteresis(image, low, high):
    weak = 50
    strong = 255
    size = image.shape
    result = np.zeros(size)
    weak_x, weak_y = np.where((image > low) & (image <= high))
    strong_x, strong_y = np.where(image >= high)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
    dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
    size = image.shape

    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if((0 <= new_x < size[0] and 0 <= new_y < size[1]) and (result[new_x, new_y] == weak)):
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    result[result != strong] = 0
    return result


def process_edge(edge):
    edge = edge.astype(np.float64)
    dx = cv2.Sobel(edge, cv2.CV_64F, 1, 0, ksize=5)
    ddx = cv2.Sobel(dx, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(edge, cv2.CV_64F, 0, 1, ksize=5)
    ddy = cv2.Sobel(dy, cv2.CV_64F, 0, 1, ksize=5)

    angles = np.rad2deg(np.arctan2(ddy, ddx))
    angles[angles < 0] += 180

    edge = non_maximum_suppression(edge, angles)
    edge = double_threshold_hysteresis(edge, 100, 150) == 255

    edge = remove_small_objects(edge, min_size=5, connectivity=2)

    return ((1 - edge) * 255).astype(np.uint8)


def get_edge(edge_net, image, device):
    return process_edge(edge_predict(edge_net, image, device))


def symmetric_chamfer(edge0, edge1):
    dist0 = cv2.distanceTransform(edge0, cv2.DIST_L2, 5)
    dist1 = cv2.distanceTransform(edge1, cv2.DIST_L2, 5)
    score = 0.5 * (dist0[edge1 == 0].mean() + dist1[edge0 == 0].mean())
    return score
