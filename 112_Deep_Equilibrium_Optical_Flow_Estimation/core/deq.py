import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

from gma import Attention

from lib.solvers import anderson, broyden
from lib.grad import make_pair, backward_factory
from lib.jacobian import power_method

from termcolor import colored 
import torch.autograd as autograd

from metrics import process_metrics


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class DEQFlowBase(nn.Module):
    def __init__(self, args):
        super(DEQFlowBase, self).__init__()
        self.args = args
        
        odim = 256
        args.corr_levels = 4
        args.corr_radius = 4

        if args.small:
            odim = 128
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_radius = 3
        elif args.large:
            self.hidden_dim = hdim = 192
            self.context_dim = cdim = 192
        elif args.huge:
            self.hidden_dim = hdim = 256
            self.context_dim = cdim = 256
        elif args.gigantic:
            self.hidden_dim = hdim = 384
            self.context_dim = cdim = 384
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=odim, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=odim, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
            
            if args.gma:
                self.attn = Attention(dim=cdim, heads=1, max_pos_size=160, dim_head=cdim)
            else:
                self.attn = None

        # Added the following for the DEQ models
        if args.wnorm:
            self.update_block._wnorm()

        self.f_solver = eval(args.f_solver)
        self.f_thres = args.f_thres
        self.eval_f_thres = int(self.f_thres * args.eval_factor)
        self.stop_mode = args.stop_mode

        self.hook = None

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    
    def _log_convergence(self, result, name="FORWARD", color="yellow"):
        stop_mode = self.stop_mode
        alt_mode = "rel" if stop_mode == "abs" else "abs"
        diff_trace, alt_diff_trace = result[f'{stop_mode}_trace'], result[f'{alt_mode}_trace']
        stop_diff, alt_diff = min(diff_trace), min(alt_diff_trace)
        print(colored(f"{'TRAIN' if self.training else 'VALID'} | {name} {stop_mode}_diff: {stop_diff}; {alt_mode}_diff: {alt_diff}; nstep: {result['nstep']}", f"{color}"))

    def _sradius(self, func, z_star):
        with torch.enable_grad():
            new_z_star = func(z_star.requires_grad_())
        _, sradius = power_method(new_z_star, z_star, n_iters=75)

        return sradius

    def _upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)
        
    def _decode(self, z_out, vec2list, coords0):
        flow_predictions = []

        for z_pred in z_out:
            net, coords1 = vec2list(z_pred)
            up_mask = .25 * self.update_block.mask(net)
            
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self._upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)
        
        return flow_predictions
    
    def _fixed_point_solve(self, deq_func, z_star, 
            seed=None, f_thres=None, 
            **kwargs):
        raise NotImplementedError
    
    def _deq_forward(self, deq_func, list2vec, vec2list, z_star, coords0,
            flow_gt=None, valid=None, step_seq_loss=None, 
            seed=None, sradius_mode=False, 
            **kwargs):
        raise NotImplementedError

    def forward(self, image1, image2, 
            flow_gt=None, valid=None, step_seq_loss=None, 
            flow_init=None, cached_result=None, sradius_mode=False, 
            **kwargs):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            # cnet = self.cnet(image1)
            # net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            # net = torch.tanh(net)
            inp = self.cnet(image1)
            inp = torch.relu(inp)

            if self.attn:
                attn = self.attn(inp)
            else:
                attn = None

        bsz, _, H, W = inp.shape
        coords0, coords1 = self._initialize_flow(image1)
        net = torch.zeros(bsz, hdim, H, W, device=inp.device)
        if cached_result:
            net, flow_pred_prev = cached_result
            coords1 = coords0 + flow_pred_prev

        if flow_init is not None:
            coords1 = coords1 + flow_init

        seed = (inp.get_device() == 0 and np.random.uniform(0,1) < 2e-3)

        def list2vec(h, c):  # h is net, c is coords1
            return torch.cat([h.view(bsz, h.shape[1], -1), c.view(bsz, c.shape[1], -1)], dim=1)

        def vec2list(hidden):
            return hidden[:,:net.shape[1]].view_as(net), hidden[:,net.shape[1]:].view_as(coords1)

        def deq_func(hidden):
            h, c = vec2list(hidden) 
            c = c.detach()

            with autocast(enabled=self.args.mixed_precision):                   
                new_h, delta_flow = self.update_block(h, inp, corr_fn(c), c-coords0, attn) # corr_fn(coords1) produces the index correlation volumes
            new_c = c + delta_flow  # F(t+1) = F(t) + \Delta(t)
            return list2vec(new_h, new_c)

        self.update_block.reset()   # In case we use weight normalization, we need to recompute the weight with wg and wv
        z_star = list2vec(net, coords1)
        
        return self._deq_forward(deq_func, list2vec, vec2list, z_star, coords0,
                flow_gt, valid, step_seq_loss, seed, sradius_mode, 
                **kwargs)


class DEQFlowIndexing(DEQFlowBase):
    def __init__(self, args):
        super(DEQFlowIndexing, self).__init__(args)
        
        # Define gradient functions through the backward factory
        if args.n_losses > 1:
            n_losses = min(args.f_thres, args.n_losses)
            delta = int(args.f_thres // n_losses)
            self.indexing = [(k+1)*delta for k in range(n_losses)]
        else:
            self.indexing = [*args.indexing, args.f_thres]
        
        # By default, we use the same phantom grad for all corrections.
        # You can also set different grad steps a, b, and c for different terms by ``args.phantom_grad a b c ...''.
        indexing_pg = make_pair(self.indexing, args.phantom_grad)
        produce_grad = [
                backward_factory(grad_type=pg, tau=args.tau, sup_all=args.sup_all) for pg in indexing_pg
                ]
        if args.ift:
            # Enabling args.ift will replace the last gradient function by IFT.
            produce_grad[-1] = backward_factory(
                grad_type='ift', safe_ift=args.safe_ift, b_solver=eval(args.b_solver),
                b_solver_kwargs=dict(threshold=args.b_thres, stop_mode=args.stop_mode)
                )

        self.produce_grad = produce_grad

    def _fixed_point_solve(self, deq_func, z_star, 
            seed=None, f_thres=None, **kwargs):
        if f_thres is None: f_thres = self.f_thres
        indexing = self.indexing if self.training else None

        with torch.no_grad():
            result = self.f_solver(deq_func, x0=z_star, threshold=f_thres, # To reuse previous coarse fixed points
                    eps=(1e-3 if self.stop_mode == "abs" else 1e-6), stop_mode=self.stop_mode, indexing=indexing)

            z_star, trajectory = result['result'], result['indexing']
        if seed: self._log_convergence(result, name="FORWARD", color="yellow")          
        
        return z_star, trajectory, min(result['rel_trace']), min(result['abs_trace'])

    def _deq_forward(self, deq_func, list2vec, vec2list, z_star, coords0,
            flow_gt=None, valid=None, step_seq_loss=None, 
            seed=None, sradius_mode=False, 
            **kwargs):
        # The code for DEQ version, where we use a wrapper. 
        if self.training:
            _, trajectory, rel_error, abs_error = self._fixed_point_solve(deq_func, z_star, seed=seed, *kwargs)
            
            z_out = []
            for z_pred, produce_grad in zip(trajectory, self.produce_grad):
                z_out += produce_grad(self, z_pred, deq_func)  # See lib/grad.py for the backward pass implementations
            
            flow_predictions = self._decode(z_out, vec2list, coords0)
            
            flow_loss, epe = step_seq_loss(flow_predictions, flow_gt, valid)
            metrics = process_metrics(epe, rel_error, abs_error)

            return flow_loss, metrics
        else:
            # During inference, we directly solve for fixed point
            z_star, _, rel_error, abs_error = self._fixed_point_solve(deq_func, z_star, seed=seed, f_thres=self.eval_f_thres)
            sradius = self._sradius(deq_func, z_star) if sradius_mode else torch.zeros(1, device=z_star.device)

            flow_up = self._decode([z_star], vec2list, coords0)[0]
            net, coords1 = vec2list(z_star)

            return coords1 - coords0, flow_up, {"sradius": sradius, "cached_result": (net, coords1 - coords0)}


class DEQFlowSliced(DEQFlowBase):
    def __init__(self, args):
        super(DEQFlowSliced, self).__init__(args)
        
        # Define gradient functions through the backward factory
        if args.n_losses > 1:
            self.indexing = [int(args.f_thres // args.n_losses) for _ in range(args.n_losses)]
        else:
            self.indexing = np.diff([0, *args.indexing, args.f_thres]).tolist()
        
        # By default, we use the same phantom grad for all corrections.
        # You can also set different grad steps a, b, and c for different terms by ``args.phantom_grad a b c ...''.
        indexing_pg = make_pair(self.indexing, args.phantom_grad)
        produce_grad = [
                backward_factory(grad_type=pg, tau=args.tau, sup_all=args.sup_all) for pg in indexing_pg
                ]
        if args.ift:
            # Enabling args.ift will replace the last gradient function by IFT.
            produce_grad[-1] = backward_factory(
                grad_type='ift', safe_ift=args.safe_ift, b_solver=eval(args.b_solver),
                b_solver_kwargs=dict(threshold=args.b_thres, stop_mode=args.stop_mode)
                )

        self.produce_grad = produce_grad
    
    def _fixed_point_solve(self, deq_func, z_star, 
            seed=None, f_thres=None, **kwargs):
        with torch.no_grad():
            result = self.f_solver(deq_func, x0=z_star, threshold=f_thres, # To reuse previous coarse fixed points
                    eps=(1e-3 if self.stop_mode == "abs" else 1e-6), stop_mode=self.stop_mode)

            z_star = result['result'] 
        if seed: self._log_convergence(result, name="FORWARD", color="yellow")          
        
        return z_star, min(result['rel_trace']), min(result['abs_trace'])

    def _deq_forward(self, deq_func, list2vec, vec2list, z_star, coords0,
            flow_gt=None, valid=None, step_seq_loss=None, 
            seed=None, sradius_mode=False, 
            **kwargs):
        # The code for DEQ version, where we use a wrapper. 
        if self.training:
            z_out = []
            for f_thres, produce_grad in zip(self.indexing, self.produce_grad):
                z_star, rel_error, abs_error = self._fixed_point_solve(deq_func, z_star, f_thres=f_thres, seed=seed)
                z_out += produce_grad(self, z_star, deq_func) # See lib/grad.py for implementations

            flow_predictions = self._decode(z_out, vec2list, coords0)

            flow_loss, epe = step_seq_loss(flow_predictions, flow_gt, valid)
            metrics = process_metrics(epe, rel_error, abs_error)

            return flow_loss, metrics
        else:
            # During inference, we directly solve for fixed point
            z_star, rel_error, abs_error = self._fixed_point_solve(deq_func, z_star, f_thres=self.eval_f_thres, seed=seed)
            sradius = self._sradius(deq_func, z_star) if sradius_mode else torch.zeros(1, device=z_star.device)
            
            flow_up = self._decode([z_star], vec2list, coords0)[0]
            net, coords1 = vec2list(z_star)

            return coords1 - coords0, flow_up, {"sradius": sradius, "cached_result": (net, coords1 - coords0)}


def get_model(args):
    if args.sliced_core:
        return DEQFlowSliced
    else:
        return DEQFlowIndexing
