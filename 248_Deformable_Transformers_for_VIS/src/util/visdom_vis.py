import copy
import logging
import torch
from visdom import Visdom

logging.getLogger('visdom').setLevel(logging.CRITICAL)


class BaseVis(object):

    def __init__(self, viz_opts, update_mode='append', env=None, win=None,
                 resume=False, port=8097, server='http://localhost'):
        self.viz_opts = viz_opts
        self.update_mode = update_mode
        self.win = win
        if env is None:
            env = 'main'
        self.viz = Visdom(env=env, port=port, server=server)
        # if resume first plot should not update with replace
        self.removed = not resume

    def win_exists(self):
        return self.viz.win_exists(self.win)

    def close(self):
        if self.win is not None:
            self.viz.close(win=self.win)
            self.win = None

    def register_event_handler(self, handler):
        self.viz.register_event_handler(handler, self.win)


class LineVis(BaseVis):
    """Visdom Line Visualization Helper Class."""

    def plot(self, y_data, x_label):
        """Plot given data.

        Appends new data to exisiting line visualization.
        """
        update = self.update_mode
        # update mode must be None the first time or after plot data was removed
        if self.removed:
            update = None
            self.removed = False

        if isinstance(x_label, list):
            Y = torch.Tensor(y_data)
            X = torch.Tensor(x_label)
        else:
            y_data = [d.cpu() if torch.is_tensor(d)
                      else torch.tensor(d)
                      for d in y_data]

            Y = torch.Tensor(y_data).unsqueeze(dim=0)
            X = torch.Tensor([x_label])

        win = self.viz.line(X=X, Y=Y, opts=self.viz_opts, win=self.win, update=update)

        if self.win is None:
            self.win = win
        self.viz.save([self.viz.env])

    def reset(self):
        #TODO: currently reset does not empty directly only on the next plot.
        # update='remove' is not working as expected.
        if self.win is not None:
            # self.viz.line(X=None, Y=None, win=self.win, update='remove')
            self.removed = True


class ImgVis(BaseVis):
    """Visdom Image Visualization Helper Class."""

    def plot(self, images):
        """Plot given images."""

        # images = [img.data if isinstance(img, torch.autograd.Variable)
        #           else img for img in images]
        # images = [img.squeeze(dim=0) if len(img.size()) == 4
        #           else img for img in images]

        self.win = self.viz.images(
            images,
            nrow=1,
            opts=self.viz_opts,
            win=self.win, )
        self.viz.save([self.viz.env])




def build_visualizers(cfg):
    visualizers = {}
    visualizers['train'] = {}
    visualizers['val'] = {}

    if not cfg.VISDOM_ON:
        return visualizers

    env_name = str(cfg.OUTPUT_DIR).split('/')[-1]

    vis_kwargs = {
        'env': env_name,
        'resume': cfg.RESUME_VIS,
        'port': cfg.VISDOM_PORT,
        'server': cfg.VISDOM_SERVER}

    #
    # METRICS
    #
    legend = [
        'class_error',
        'loss',
        'loss_bbox',
        'loss_ce',
        'loss_giou',
        'loss_mask',
        'loss_dice',
        'loss_bbox_unscaled',
        'loss_ce_unscaled',
        'loss_giou_unscaled',
        'loss_mask_unscaled',
        'loss_dice_unscaled',
        'lr_base',
        'lr_backbone',
        'lr_linear_proj',
        'lr_mask_head',
        'lr_temporal_linear_proj',
        'iter_time'
    ]

    if not cfg.DATASETS.TYPE == 'vis' and not cfg.MODEL.MASK_ON:
        legend.remove('loss_mask')
        legend.remove('loss_mask_unscaled')
        legend.remove('loss_dice')
        legend.remove('loss_dice_unscaled')


    opts = dict(
        title="TRAIN METRICS ITERS",
        xlabel='ITERS',
        ylabel='METRICS',
        width=1000,
        height=500,
        legend=legend)

    # TRAIN
    visualizers['train']['iter_metrics'] = LineVis(opts, **vis_kwargs)

    if cfg.DATASETS.TYPE == 'vis':
        return visualizers

    # TODO: Implement visdom for VIS when GT is available for the val dataset
    opts = copy.deepcopy(opts)
    opts['title'] = "VAL METRICS EPOCHS"
    opts['xlabel'] = "EPOCHS"
    opts['legend'].remove('lr_base')
    opts['legend'].remove('lr_backbone')
    opts['legend'].remove('lr_linear_proj')
    opts['legend'].remove('lr_mask_head')
    opts['legend'].remove('lr_temporal_linear_proj')
    opts['legend'].remove('iter_time')
    visualizers['val']['epoch_metrics'] = LineVis(opts, **vis_kwargs)

    # EVAL COCO

    legend = [
        'BBOX AP IoU=0.50:0.95',
        'BBOX AP IoU=0.50',
        'BBOX AP IoU=0.75',
    ]

    if cfg.MODEL.MASK_ON:
        legend.extend([
            'MASK AP IoU=0.50:0.95',
            'MASK AP IoU=0.50',
            'MASK AP IoU=0.75'])

    opts = dict(
        title='VAL EVAL EPOCHS',
        xlabel='EPOCHS',
        ylabel='METRICS',
        width=1000,
        height=500,
        legend=legend)

    opts['title'] = 'VAL EVAL EPOCHS'
    visualizers['val']['epoch_eval'] = LineVis(opts, **vis_kwargs)

    return visualizers

def get_vis_win_names(vis_dict):
    vis_win_names = {
        outer_k: {
            inner_k: inner_v.win
            for inner_k, inner_v in outer_v.items()
        }
        for outer_k, outer_v in vis_dict.items()
    }
    return vis_win_names
