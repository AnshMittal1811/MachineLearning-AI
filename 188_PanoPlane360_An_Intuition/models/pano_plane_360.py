import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.ndimage.filters import maximum_filter

from . import backbone
from . import models_utils

class Net(nn.Module):
    def __init__(self, backbone_name='ResnetFPN', backbone_kwargs={},
                 embd_dim=2, yawinv=True):
        super(Net, self).__init__()
        self.yawinv = yawinv

        BackboneModel = getattr(backbone, backbone_name)
        self.backbone = BackboneModel(**backbone_kwargs)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 1024)
            channels = [p.shape[1] for p in self.backbone(dummy)]

        self.embd_dim = embd_dim
        self.pred_h_planar = nn.Conv2d(channels[-1], 1, kernel_size=1)
        self.pred_v_planar = nn.Conv2d(channels[-1], 1, kernel_size=1)
        self.pred_h_embd = nn.Conv2d(channels[-1], self.embd_dim, kernel_size=1)
        self.pred_v_embd = nn.Conv2d(channels[-1], self.embd_dim, kernel_size=1)
        self.pred_h_planar.bias.data.fill_(0)
        self.pred_v_planar.bias.data.fill_(0)

        geo_in_ch = 64
        self.fpn_refine = FPN_refine(fpn_ch=256, mid_out=128, final_out=geo_in_ch)
        if self.backbone.lr_pad:
            backbone.wrap_lr_pad(self.fpn_refine)

        self.pred_depth = nn.Conv2d(geo_in_ch, 1, kernel_size=1)
        self.pred_v_param_deg = nn.Conv2d(geo_in_ch, 1, kernel_size=1)
        self.pred_depth.bias.data.fill_(1.5)
        self.pred_v_param_deg.bias.data.fill_(0)

    def infer_depth(self, input_dict, args):
        assert 'preprocessed' in input_dict, 'You may forget to preprocess the input_dict'
        assert input_dict['rgb'].shape[0] == 1, 'Batch size must be 1 in test mode'
        x = input_dict['rgb']
        p0, p1, p2, p3, p4, p5 = self.backbone(x)
        geo_feature = self.fpn_refine([F.max_pool2d(p5, 2), p5, p4, p3, p2])
        depth = F.interpolate(
                    self.pred_depth(geo_feature),
                    scale_factor=2, mode='bilinear', align_corners=True)
        return {'depth': depth[0,0].cpu().numpy()}

    def get_v_loss(self, input_dict, args):
        with torch.no_grad():
            rgb = input_dict['rgb']
            mask = ~input_dict['dontcare']
            vplane = input_dict['vplane']
            v_planar = (vplane[:,:2].abs().sum(1, keepdim=True) != 0)   # batch X 1 X H X W
            v_param = models_utils.vplane_2_vparam(vplane)

            output_dict = self.forward(rgb)
            #vpred = output_dict['v_param'].permute(0,2,3,1)[v_planar[:, 0]]  # *, 2
            vtarget = v_param.permute(0,2,3,1)[v_planar[:, 0]]               # *, 2
            #losses['v_param'] = (vpred - vtarget).abs().mean()
            deg_pred = output_dict['v_param_deg'].permute(0,2,3,1)[v_planar[:, 0]]  # *, 2
            deg_target = torch.atan2(vtarget[:, [1]], vtarget[:, [0]])              # *, 2
            _, deg = radius_loss(deg_pred, deg_target)
        return deg

    def forward(self, x):
        p0, p1, p2, p3, p4, p5 = self.backbone(x)
        output_dict = {
            'h_planar': self.pred_h_planar(p0),
            'v_planar': self.pred_v_planar(p0),
            'h_embedding': self.pred_h_embd(p0),
            'v_embedding': self.pred_v_embd(p0),
        }
        geo_feature = self.fpn_refine([F.max_pool2d(p5, 2), p5, p4, p3, p2])
        depth = F.interpolate(
            self.pred_depth(geo_feature),
            scale_factor=2, mode='bilinear', align_corners=True)
        Q = models_utils.depth_2_Q(depth)  # depth to XYZ in [B, 3, H_, W]
        v_param_deg = F.interpolate(
            self.pred_v_param_deg(geo_feature),
            scale_factor=2, mode='bilinear', align_corners=True)
        if self.yawinv:
            u_grid = models_utils.u_grid(*v_param_deg.shape)
            v_param_deg = v_param_deg + u_grid.to(v_param_deg)

        v_param = torch.cat([
            torch.cos(v_param_deg), torch.sin(v_param_deg),
        ], dim=1)
        v_param = v_param * (v_param * Q[:, :2]).sum(dim=1, keepdim=True).clamp(min=1e-4)

        output_dict['depth'] = depth
        output_dict['h_param'] = Q[:, [2]]
        output_dict['v_param'] = v_param
        output_dict['v_param_deg'] = v_param_deg
        return output_dict

    def infer_HVmap(self, input_dict, args):
        assert 'preprocessed' in input_dict, 'You may forget to preprocess the input_dict'
        assert input_dict['rgb'].shape[0] == 1, 'Batch size must be 1 in test mode'
        rgb = input_dict['rgb']
        H_, W = rgb.shape[2:]
        with torch.no_grad():
            output_dict = self.forward(rgb)
            pred_h_param = output_dict['h_param']
            pred_v_param = output_dict['v_param']
            output_dict['h_param'] = pred_h_param
            output_dict['v_param'] = pred_v_param
            h_planar_prob = torch.sigmoid(output_dict['h_planar'])
            v_planar_prob = torch.sigmoid(output_dict['v_planar'])
            h_planar_prob[h_planar_prob < v_planar_prob] = 0
            v_planar_prob[v_planar_prob < h_planar_prob] = 0
            h_planar_mask = (h_planar_prob > 0.4)
            v_planar_mask = (v_planar_prob > 0.4)

            # Mean shift segmentation
            bin_mean_shift = Bin_Mean_Shift(device=args.device)
            seg_H, _, _ = bin_mean_shift.test_forward(
                h_planar_prob[0],
                output_dict['h_embedding'][0],
                output_dict['h_param'][0],
                mask_threshold=0.4)
            seg_H = seg_H.argmax(1).reshape(H_, W)

            seg_V = torch.zeros_like(seg_H) - 1
            seg_v_pre = segement_by_rad(v_planar_mask[0,0], output_dict['v_param_deg'][0,0])
            #seg_v_pre = v_planar_mask[0,0].long()
            for i in range(1, 1+seg_v_pre.max()):
                cur_mask = (seg_v_pre==i)
                cur_prob = torch.where(
                    cur_mask,
                    v_planar_prob[0,0],
                    torch.zeros_like(v_planar_prob[0,0]),
                )
                cur_seg_V, _, _ = bin_mean_shift.test_forward(
                    cur_prob,
                    output_dict['v_embedding'][0],
                    output_dict['v_param'][0],
                    mask_threshold=0.4)
                cur_seg_V = cur_seg_V.argmax(1).reshape(H_, W)
                seg_V[cur_mask] = seg_V.max() + 1 + cur_seg_V[cur_mask]

            infer_dict = {}
            # Construct h_plane
            h_planes = np.zeros([H_, W], np.float32)
            h_param = output_dict['h_param'][0,0]  # HW
            for i in range(seg_H.max().item() + 1):
                cur_mask = ((seg_H == i) & h_planar_mask[0,0]) # HW
                if cur_mask.sum() < 100:
                    continue
                cur_param = h_param[cur_mask].median()
                h_planes[cur_mask.cpu().numpy()] = cur_param.item()
            infer_dict['h_planes'] = h_planes

            # Construct v_plane
            v_param = output_dict['v_param'].clone()  # 12HW
            v_planar_mask_upd = torch.zeros_like(v_planar_mask[0,0])
            for i in range(seg_V.max().item() + 1):
                cur_mask = ((seg_V == i) & v_planar_mask[0,0]) # 11HW
                if cur_mask.sum() < 100:
                    continue
                v_planar_mask_upd |= cur_mask
                x = v_param[0,0][cur_mask].median()
                y = v_param[0,1][cur_mask].median()
                v_param[0,0][cur_mask] = x
                v_param[0,1][cur_mask] = y
            v_planes = models_utils.vparam_2_vplane(v_param) # 13HW
            v_planes[~v_planar_mask.repeat(1,3,1,1)] = 0
            v_planes[~v_planar_mask_upd.repeat(1,3,1,1)] = 0
            infer_dict['v_planes'] = v_planes[0].permute(1,2,0).cpu().numpy()

        '''
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('jet')
        seg_H = seg_H.cpu().numpy()
        seg_V = seg_V.cpu().numpy()
        seg_H_vis = np.unique(seg_H, return_inverse=True)[1].reshape(H_, W)
        seg_H_vis = cmap(seg_H_vis/seg_H_vis.max())[..., :3]
        seg_H_vis[seg_H < 0] = 0
        seg_V_vis = np.unique(seg_V, return_inverse=True)[1].reshape(H_, W)
        seg_V_vis = cmap(seg_V_vis/seg_V_vis.max())[..., :3]
        seg_V_vis[seg_V < 0] = 0
        seg_v_pre = seg_v_pre.cpu().numpy()
        seg_v_pre_vis = cmap(seg_v_pre/seg_v_pre.max())[..., :3]
        seg_v_pre_vis[seg_V < 0] = 0
        infer_dict['seg_H'] = (seg_H_vis * 255).astype(np.uint8)
        infer_dict['seg_V'] = (seg_V_vis * 255).astype(np.uint8)
        infer_dict['seg_pre'] = (seg_v_pre_vis * 255).astype(np.uint8)
        del infer_dict['v_planes'], infer_dict['h_planes']
        '''

        return infer_dict

    def compute_losses(self, input_dict, args):
        assert 'preprocessed' in input_dict, 'You may forget to preprocess the input_dict'
        rgb = input_dict['rgb']
        depth = input_dict['depth']
        mask = ~input_dict['dontcare']
        hplane = input_dict['hplane']
        vplane = input_dict['vplane']
        output_dict = self.forward(rgb)
        losses = {}

        # H/V Planar loss
        h_planar = (hplane != 0)                                    # not 0 is h, else not h
        v_planar = (vplane[:,:2].abs().sum(1, keepdim=True) != 0)   # batch X 1 X H X W
        losses['h_planar'] = F.binary_cross_entropy_with_logits(
            output_dict['h_planar'][mask],
            h_planar[mask].float()
        )
        losses['v_planar'] = F.binary_cross_entropy_with_logits(
            output_dict['v_planar'][mask],
            v_planar[mask].float()
        )

        # Per-pixel parameter loss
        losses['h_param'] = (output_dict['h_param'][h_planar] - hplane[h_planar]).abs().mean()
        v_param = models_utils.vplane_2_vparam(vplane)
        losses['depth'] = (output_dict['depth'] - depth)[mask].abs().mean()
        vpred = output_dict['v_param'].permute(0,2,3,1)[v_planar[:, 0]]  # *, 2
        vtarget = v_param.permute(0,2,3,1)[v_planar[:, 0]]               # *, 2
        losses['v_param'] = (vpred - vtarget).abs().mean()
        deg_pred = output_dict['v_param_deg'].permute(0,2,3,1)[v_planar[:, 0]]  # *, 2
        deg_target = torch.atan2(vtarget[:, [1]], vtarget[:, [0]])              # *, 2
        losses['v_param_rad'], losses['v_param_deg'] = radius_loss(deg_pred, deg_target)
        losses['v_param_total'] = losses['v_param'] + losses['v_param_rad']

        # Per-instance parameter loss
        losses['h_instance_param'] = torch.zeros_like(losses['h_planar'])
        losses['v_instance_param'] = torch.zeros_like(losses['v_planar'])

        # Pull/Push loss for embedding
        losses['h_pull'], losses['h_push'] = hinge_embedding_loss(output_dict['h_embedding'], hplane)
        losses['v_pull'], losses['v_push'] = hinge_embedding_loss(output_dict['v_embedding'], vplane)

        # Final
        losses['total'] = 0.1*losses['h_planar'] + 0.1*losses['v_planar'] + \
                          losses['h_param'] + losses['v_param_total'] + \
                          losses['h_pull'] + losses['h_push'] + losses['v_pull'] + losses['v_push'] + \
                          losses['h_instance_param'] + losses['v_instance_param']
        if 'depth' in losses:
            losses['total'] = losses['total'] + losses['depth']
        with torch.no_grad():
            losses['h_planar_acc'] = (h_planar == (output_dict['h_planar'] > 0))[mask].float().mean()
            losses['v_planar_acc'] = (v_planar == (output_dict['v_planar'] > 0))[mask].float().mean()
            losses['print/h_dis'] = losses['h_param']
            losses['print/v_dis'] = (output_dict['v_param'] - v_param).norm(p=2, dim=1)[v_planar[:,0]].mean()
            losses['print/v_deg'] = losses['v_param_deg']
            losses['print/h_planar_acc'] = losses['h_planar_acc']
            losses['print/v_planar_acc'] = losses['v_planar_acc']
            losses['print/h_pull'] = losses['h_pull']
            losses['print/h_push'] = losses['h_push']
            losses['print/v_pull'] = losses['v_pull']
            losses['print/v_push'] = losses['v_push']
            if losses['h_pull'] == 0: del losses['h_pull']
            if losses['h_push'] == 0: del losses['h_push']
            if losses['v_pull'] == 0: del losses['v_pull']
            if losses['v_push'] == 0: del losses['v_push']
        return losses

def radius_loss(radA, radB):
    radA_xy = torch.stack([torch.cos(radA), torch.sin(radA)], 0)
    radB_xy = torch.stack([torch.cos(radB), torch.sin(radB)], 0)
    cos_sim = (radA_xy * radB_xy).sum(dim=0)
    cos_loss = (1 - cos_sim).mean()
    degree = cos_sim.clamp(-1, 1).acos().mean() / np.pi * 180
    return cos_loss, degree

def find_peaks(signal, mode='wrap', radius=29, mincnt=100):
    max_v = maximum_filter(signal, size=radius, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > mincnt]
    return pk_loc

def segement_by_rad(mask, rad, binsz=1/360, radius=25+25+1, mincnt=100):
    # mask:  [H, W]  binary mask
    # rad:   [H, W]  radius for voting
    device = rad.device
    mask = mask.cpu().numpy()
    rad = rad.cpu().numpy() % (np.pi * 2)
    votebin = np.bincount((rad[mask] / (2*np.pi) / binsz).astype(int))
    pk_loc = find_peaks(votebin, mode='wrap', radius=radius, mincnt=mincnt)
    if len(pk_loc):
        pk_rad = pk_loc * 2 * np.pi / 360
        xy2d = np.stack([np.cos(rad), np.sin(rad)], -1).reshape(-1,2)
        px2d = np.stack([np.cos(pk_rad), np.sin(pk_rad)], 0)
        seg = 1 + (xy2d @ px2d).argmax(-1).reshape(rad.shape)
        seg[~mask] = 0
    else:
        print('[Warning] no center vote by rad')
        seg = np.zeros(rad.shape, np.int32)
    return torch.from_numpy(seg).to(device)

# Below codes are adapted from: https://github.com/svip-lab/PlanarReconstruction/
def hinge_embedding_loss(embedding, plane, t_pull=0.5, t_push=1.5):
    device = embedding.device
    b, c, h, w = embedding.shape

    pull_loss = 0
    push_loss = 0
    for ith in range(b):
        cur_embedding = embedding[ith].permute(1, 2, 0)  # H,W,dim
        with torch.no_grad():
            planes_mask, planes_params = models_utils.plane_seperation(plane[ith])
        n_planes = planes_mask.shape[0]

        # Intra-embedding pull loss
        centroids = []
        for one_plane_mask in planes_mask:
            embd_pts = cur_embedding[one_plane_mask]
            embd_cen = embd_pts.mean(0)
            centroids.append(embd_cen)
            dis = torch.norm(embd_pts - embd_cen[None, :], p=2, dim=1) - t_pull
            pull_loss = pull_loss + F.relu(dis).mean() / (b * n_planes)

        if n_planes == 1:
            continue

        # Inter-embedding push loss
        centroids = torch.stack(centroids, 0)
        pair_dis = torch.norm(centroids[:, None] - centroids, p=2, dim=2)
        dis = t_push - pair_dis[torch.eye(n_planes).to(device) == 0]
        push_loss = push_loss + F.relu(dis).mean() / b

    return pull_loss, push_loss

# Below codes are adapted from: https://github.com/svip-lab/PlanarReconstruction/
class Bin_Mean_Shift(nn.Module):
    def __init__(self, train_iter=5, test_iter=30, bandwidth=0.5, device='cpu'):
        super(Bin_Mean_Shift, self).__init__()
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.bandwidth = bandwidth / 2.
        self.anchor_num = 20     # Rescale from 10    (the numbers of planar pixel is ~3.6 time more)
        self.sample_num = 11000  # Rescale from 3000  (the numbers of planar pixel is ~3.6 time more)
        self.device = device

    def generate_seed(self, point, bin_num):
        """
        :param point: tensor of size (K, 2)
        :param bin_num: int
        :return: seed_point
        """
        def get_start_end(a, b, k):
            start = a + (b - a) / ((k + 1) * 2)
            end = b - (b - a) / ((k + 1) * 2)
            return start, end

        min_x, min_y = point.min(dim=0)[0]
        max_x, max_y = point.max(dim=0)[0]

        start_x, end_x = get_start_end(min_x.item(), max_x.item(), bin_num)
        start_y, end_y = get_start_end(min_y.item(), max_y.item(), bin_num)

        x = torch.linspace(start_x, end_x, bin_num).view(bin_num, 1)
        y = torch.linspace(start_y, end_y, bin_num).view(1, bin_num)

        x_repeat = x.repeat(1, bin_num).view(-1, 1)
        y_repeat = y.repeat(bin_num, 1).view(-1, 1)

        return torch.cat((x_repeat, y_repeat), dim=1).to(self.device)

    def filter_seed(self, point, prob, seed_point, bandwidth, min_count=3):
        """
        :param point: tensor of size (K, 2)
        :param seed_point: tensor of size (n, 2)
        :param prob: tensor of size (K, 1) indicate probability of being plane
        :param min_count:  mini_count within a bandwith of seed point
        :param bandwidth: float
        :return: filtered_seed_points
        """
        distance_matrix = self.cal_distance_matrix(seed_point, point)  # (n, K)
        thres_matrix = (distance_matrix < bandwidth).type(torch.float32) * prob.t()
        count = thres_matrix.sum(dim=1)                  # (n, 1)
        valid = count > min_count
        return seed_point[valid]

    def cal_distance_matrix(self, point_a, point_b):
        """
        :param point_a: tensor of size (m, 2)
        :param point_b: tensor of size (n, 2)
        :return: distance matrix of size (m, n)
        """
        m, n = point_a.size(0), point_b.size(0)

        a_repeat = point_a.repeat(1, n).view(n * m, 2)                  # (n*m, 2)
        b_repeat = point_b.repeat(m, 1)                                 # (n*m, 2)

        distance = torch.nn.PairwiseDistance(keepdim=True)(a_repeat, b_repeat)  # (n*m, 1)

        return distance.view(m, n)

    def shift(self, point, prob, seed_point, bandwidth):
        """
        shift seed points
        :param point: tensor of size (K, 2)
        :param seed_point: tensor of size (n, 2)
        :param prob: tensor of size (K, 1) indicate probability of being plane
        :param bandwidth: float
        :return:  shifted points with size (n, 2)
        """
        distance_matrix = self.cal_distance_matrix(seed_point, point)  # (n, K)
        kernel_matrix = torch.exp((-0.5 / bandwidth**2) * (distance_matrix ** 2)) * (1. / (bandwidth * np.sqrt(2 * np.pi)))
        weighted_matrix = kernel_matrix * prob.t()

        # normalize matrix
        normalized_matrix = weighted_matrix / (weighted_matrix.sum(dim=1, keepdim=True) + 1e-9)
        shifted_point = torch.matmul(normalized_matrix, point)  # (n, K) * (K, 2) -> (n, 2)

        return shifted_point

    def label2onehot(self, labels):
        """
        convert a label to one hot vector
        :param labels: tensor with size (n, 1)
        :return: one hot vector tensor with size (n, max_lales+1)
        """
        n = labels.size(0)
        label_num = torch.max(labels).int() + 1

        onehot = torch.zeros((n, label_num))
        onehot.scatter_(1, labels.long(), 1.)

        return onehot.to(self.device)

    def merge_center(self, seed_point, bandwidth=0.25):
        """
        merge close seed points
        :param seed_point: tensor of size (n, 2)
        :param bandwidth: float
        :return: merged center
        """
        n = seed_point.size(0)

        # 1. calculate intensity
        distance_matrix = self.cal_distance_matrix(seed_point, seed_point)  # (n, n)
        intensity = (distance_matrix < bandwidth).type(torch.float32).sum(dim=1)

        # merge center if distance between two points less than bandwidth
        sorted_intensity, indices = torch.sort(intensity, descending=True)
        is_center = np.ones(n, dtype=np.bool)
        indices = indices.cpu().numpy()
        center = np.zeros(n, dtype=np.uint8)

        labels = np.zeros(n, dtype=np.int32)
        cur_label = 0
        for i in range(n):
            if is_center[i]:
                labels[indices[i]] = cur_label
                center[indices[i]] = 1
                for j in range(i + 1, n):
                    if is_center[j]:
                        if distance_matrix[indices[i], indices[j]] < bandwidth:
                            is_center[j] = 0
                            labels[indices[j]] = cur_label
                cur_label += 1
        # print(labels)
        # print(center)
        # return seed_point[torch.ByteTensor(center)]

        # change mask select to matrix multiply to select points
        one_hot = self.label2onehot(torch.Tensor(labels).view(-1, 1))  # (n, label_num)
        weight = one_hot / one_hot.sum(dim=0, keepdim=True)   # (n, label_num)

        return torch.matmul(weight.t(), seed_point)

    def cluster(self, point, center):
        """
        cluter each point to nearset center
        :param point: tensor with size (K, 2)
        :param center: tensor with size (n, 2)
        :return: clustering results, tensor with size (K, n) and sum to one for each row
        """
        # plus 0.01 to avoid divide by zero
        distance_matrix = 1. / (self.cal_distance_matrix(point, center)+0.01)  # (K, n)
        segmentation = F.softmax(distance_matrix, dim=1)
        return segmentation

    def bin_shift(self, prob, embedding, param, gt_seg, bandwidth):
        """
        discrete seeding mean shift in training stage
        :param prob: tensor with size (1, h, w) indicate probability of being plane
        :param embedding: tensor with size (2, h, w)
        :param param: tensor with size (3, h, w)
        :param gt_seg: ground truth instance segmentation, used for sampling planar embeddings
        :param bandwidth: float
        :return: segmentation results, tensor with size (h*w, K), K is cluster number, row sum to 1
                 sampled segmentation results, tensor with size (N, K) where N is sample size, K is cluster number, row sum to 1
                center, tensor with size (K, 2) cluster center in embedding space
                sample_prob, tensor with size (N, 1) sampled probability
                sample_seg, tensor with size (N, 1) sampled ground truth instance segmentation
                sample_params, tensor with size (3, N), sampled params
        """

        c, h, w = embedding.size()

        embedding = embedding.view(c, h*w).t()
        param = param.view(param.shape[0], h*w)
        prob = prob.view(h*w, 1)
        seg = gt_seg.view(-1)

        # random sample planar region data points using ground truth label to speed up training
        #rand_index = np.random.choice(np.arange(0, h * w)[seg.cpu().numpy() != 0], self.sample_num)
        rand_index = np.arange(0, h * w)[seg.cpu().numpy() != 0]

        sample_embedding = embedding[rand_index]
        sample_prob = prob[rand_index]
        sample_param = param[:, rand_index]

        # generate seed points and filter out those with low density to speed up training
        seed_point = self.generate_seed(sample_embedding, self.anchor_num)
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=3)
        if torch.numel(seed_point) <= 0:
            return None, None, None, None, None, None

        with torch.no_grad():
            for iter in range(self.train_iter):
                seed_point = self.shift(sample_embedding, sample_prob, seed_point, self.bandwidth)

        # filter again and merge seed points
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=10)
        if torch.numel(seed_point) <= 0:
            return None, None, None, None, None, None

        center = self.merge_center(seed_point, bandwidth=self.bandwidth)

        # cluster points
        segmentation = self.cluster(embedding, center)
        sampled_segmentation = segmentation[rand_index]

        return segmentation, sampled_segmentation, center, sample_prob, seg[rand_index].view(-1, 1), sample_param

    def forward(self, logit, embedding, param, gt_seg):
        batch_size, c, h, w = embedding.size()
        assert(c == 2)

        # apply mean shift to every item
        segmentations, sample_segmentations, centers, sample_probs, sample_gt_segs, sample_params = [], [], [], [], [], []
        for b in range(batch_size):
            segmentation, sample_segmentation, center, prob, sample_seg, sample_param = \
                self.bin_shift(torch.sigmoid(logit[b]), embedding[b], param[b], gt_seg[b], self.bandwidth)

            segmentations.append(segmentation)
            sample_segmentations.append(sample_segmentation)
            centers.append(center)
            sample_probs.append(prob)
            sample_gt_segs.append(sample_seg)
            sample_params.append(sample_param)

        return segmentations, sample_segmentations, sample_params, centers, sample_probs, sample_gt_segs

    def test_forward(self, prob, embedding, param, mask_threshold):
        """
        :param prob: probability of planar, tensor with size (1, h, w)
        :param embedding: tensor with size (2, h, w)
        :param mask_threshold: threshold of planar region
        :return: clustering results: numpy array with shape (h, w),
                 sampled segmentation results, tensor with size (N, K) where N is sample size, K is cluster number, row sum to 1
                 sample_params, tensor with size (3, N), sampled params
        """

        c, h, w = embedding.size()

        embedding = embedding.view(c, h*w).t()
        prob = prob.view(h*w, 1)
        param = param.view(param.shape[0], h * w)

        # random sample planar region data points
        rand_index = np.random.choice(np.arange(0, h * w)[prob.cpu().numpy().reshape(-1) > mask_threshold], self.sample_num)

        sample_embedding = embedding[rand_index]
        sample_prob = prob[rand_index]
        sample_param = param[:, rand_index]

        # generate seed points and filter out those with low density
        seed_point = self.generate_seed(sample_embedding, self.anchor_num)
        # seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=3)

        # New way to init
        seed_point = []
        maskout = prob.detach().clone()
        for i in range(self.anchor_num**2):
            if maskout.max() < mask_threshold:
                break
            idx = maskout.argmax()
            seed_point.append(embedding[idx])
            dist = (embedding - embedding[[idx]]).norm(dim=1)
            maskout[dist <= self.bandwidth] = -1
        seed_point = torch.stack(seed_point, 0).clone()
        sample_embedding = embedding[prob.reshape(-1) > mask_threshold]
        sample_prob = prob[prob.reshape(-1) > mask_threshold]

        with torch.no_grad():
            # start shift points
            for iter in range(self.test_iter):
                seed_point = self.shift(sample_embedding, sample_prob, seed_point, self.bandwidth)

        # filter again and merge seed points
        # seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=10)

        center = self.merge_center(seed_point, bandwidth=self.bandwidth)

        # cluster points using sample_embedding
        segmentation = self.cluster(embedding, center)

        sampled_segmentation = segmentation[rand_index]

        return segmentation, sampled_segmentation, sample_param


class FPN_refine(nn.Module):
    def __init__(self, fpn_ch=256, mid_out=128, final_out=64):
        super(FPN_refine, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(fpn_ch, mid_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_out, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(fpn_ch, mid_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_out, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(fpn_ch, mid_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_out, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(fpn_ch, mid_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_out, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(fpn_ch, mid_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_out, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.deconv1 = nn.Sequential(
            nn.Conv2d(mid_out, mid_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_out, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(mid_out*2, mid_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_out, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(mid_out*2, mid_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_out, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(mid_out*2, mid_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_out, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.Conv2d(mid_out*2, final_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(final_out, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature_maps):
        x = self.deconv1(
            F.interpolate(
                self.conv1(feature_maps[0]),
                size=feature_maps[1].shape[2:], mode='bilinear', align_corners=True
            )
        )
        x = self.deconv2(
            F.interpolate(
                torch.cat([self.conv2(feature_maps[1]), x], dim=1),
                size=feature_maps[2].shape[2:], mode='bilinear', align_corners=True
            )
        )
        x = self.deconv3(
            F.interpolate(
                torch.cat([self.conv3(feature_maps[2]), x], dim=1),
                size=feature_maps[3].shape[2:], mode='bilinear', align_corners=True
            )
        )
        x = self.deconv4(
            F.interpolate(
                torch.cat([self.conv4(feature_maps[3]), x], dim=1),
                size=feature_maps[4].shape[2:], mode='bilinear', align_corners=True
            )
        )
        x = self.deconv5(
            F.interpolate(
                torch.cat([self.conv5(feature_maps[4]), x], dim=1),
                scale_factor=2, mode='bilinear', align_corners=True
            )
        )

        return x

