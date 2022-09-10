import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import math
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .transformer.deformable_transformer import build_deforamble_transformer, build_deformable_encoder, build_deformable_decoder, build_transformer_decoder, SimpleDecoder, SimpleDecoderV2, OursDecoder, OursDecoderV2, OursDecoderV2_exp
from .ops.modules import MSDeformAttn
from .backbone.backbone import build_backbone
from .matcher.matcher import build_matcher
from .detr_util.detr_util import SetCriterion, PostProcess


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PTSEFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.d_encoder = build_deformable_encoder(cfg, num_encoder_layers=2)
        self.d_decoder = build_deformable_decoder(cfg, num_decoder_layers=2)
        self.pre_decoder = build_deformable_decoder(cfg, num_decoder_layers=2)  # for ref
        # self.o_decoder = build_transformer_decoder(cfg, num_decoder_layers=1)
        self.s_decoder1 = SimpleDecoderV2(num_layers=2)
        self.s_decoder2 = SimpleDecoderV2(num_layers=2)
        self.corr = OursDecoderV2(num_layers=2)
        self.our_decoder = OursDecoderV2(num_layers=2)
        # self.q_attn = None
        self.reference_points = nn.Linear(cfg.MODEL.hidden_dim, 2)
        self.reference_points1 = nn.Linear(cfg.MODEL.hidden_dim, 2)
        self.level_embed = nn.Parameter(torch.Tensor(cfg.MODEL.num_feature_levels, cfg.MODEL.hidden_dim))
        self.input_proj = self.build_input_proj()
        self.class_embed = nn.Linear(cfg.MODEL.hidden_dim, cfg.MODEL.num_classes)
        self.bbox_embed = MLP(cfg.MODEL.hidden_dim, cfg.MODEL.hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(cfg.MODEL.num_queries, cfg.MODEL.hidden_dim*2)

        self.init_paras()

        self.qln = nn.Linear(cfg.MODEL.hidden_dim, cfg.MODEL.hidden_dim*2)


        def fill_fc_weights(layers):
            for m in layers.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        fill_fc_weights(self.qln)

    def build_input_proj(self):
        if self.cfg.MODEL.num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.cfg.MODEL.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.cfg.MODEL.hidden_dim),
                ))
            for _ in range(self.cfg.MODEL.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.cfg.MODEL.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.cfg.MODEL.hidden_dim),
                ))
                in_channels = self.cfg.MODEL.hidden_dim
            input_proj = nn.ModuleList(input_proj_list)
        else:
            input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.backbone.num_channels[0], self.cfg.MODEL.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.cfg.MODEL.hidden_dim),
                )])
        return input_proj

    def init_paras(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.cfg.MODEL.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        num_pred = self.cfg.MODEL.dec_layers
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
        self.d_decoder.bbox_embed = None

        for l in [self.d_encoder.parameters(), self.d_decoder.parameters(), self.reference_points.parameters(), self.reference_points1.parameters()]:
            for p in l:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        xavier_uniform_(self.reference_points1.weight.data, gain=1.0)
        constant_(self.reference_points1.bias.data, 0.)
        normal_(self.level_embed)

    def forward(self, samples_dict: dict):
        """
        # {"cur": NestedTensor1, "ref": [NestedTensor1, NestedTensor2]}
        :param samples: NestedTensor in dict
        :return:
        """
        img_curr_nest = samples_dict['cur'] # BCHW
        # print(img_curr)
        img_ref_nest_list = samples_dict['ref_l']

        query_embeds = self.query_embed.weight  # todo

        # -------------------------------1st stage------------------------------------
        mem_ref_stg1_list = []
        mask_ref_stg1_list = []
        for nest in img_ref_nest_list:
            srcs, masks, pos = self.get_feat(nest)  # srcs here are list, not flattend yet, downsampling by
            memory, dec_utils = self.d_enc(srcs, masks, pos)
            spatial_shapes, level_start_index, valid_ratios, mask_flatten = dec_utils
            # print(sum(sum(mask_flatten==True)))
            memory_level_list = []
            mask_level_list = []
            for i in range(len(level_start_index) - 1):
                memory_l = memory[:, level_start_index[i]:level_start_index[i + 1], :]
                memory_level_list.append(memory_l)
                mask_l = mask_flatten[:, level_start_index[i]:level_start_index[i + 1]]
                mask_level_list.append(mask_l)
            memory_level_list.append(memory[:, level_start_index[-1]:, :])
            mask_level_list.append(mask_flatten[:, level_start_index[-1]:])

            mem_ref_stg1_list.append(memory_level_list)
            mask_ref_stg1_list.append(mask_level_list)

        mem_ref_stg1_cat = [torch.cat(item, dim=1) for item in list(zip(*mem_ref_stg1_list))]
        mask_ref_stg1_cat = [torch.cat(item, dim=1) for item in list(zip(*mask_ref_stg1_list))]

        mem_ref_stg1_cat4q = [torch.cat(item, dim=1) for item in mem_ref_stg1_list]
        # for cur
        srcs, masks, pos = self.get_feat(img_curr_nest)

        memory, dec_utils = self.d_enc(srcs, masks, pos)
        spatial_shapes, level_start_index, valid_ratios, mask_flatten = dec_utils
        # print(sum(sum(mask_flatten == True)))
        mem_cur_stg1 = []
        mask_cur_stg1 = []
        for i in range(len(level_start_index) - 1):
            memory_l = memory[:, level_start_index[i]:level_start_index[i + 1], :]
            mem_cur_stg1.append(memory_l)
            mask_l = mask_flatten[:, level_start_index[i]:level_start_index[i + 1]]
            mask_cur_stg1.append(mask_l)
        mem_cur_stg1.append(memory[:, level_start_index[-1]:, :])
        mask_cur_stg1.append(mask_flatten[:, level_start_index[-1]:])

        # -------------------------------2nd stage------------------------------------
        mem_ref_stg2_list = []
        for mem_ref, mask_ref in zip(mem_ref_stg1_list, mask_ref_stg1_list):
            mem_ref_stg2_level_list = []
            for mem_ref_level, mask_ref_level, mem_cur_level, mask_cur_level in zip(mem_ref, mask_ref, mem_cur_stg1, mask_cur_stg1):
                mem_ref_stg2 = self.corr(tgt=mem_ref_level, tgt_mask=None, memory=mem_cur_level, memory_mask=None).squeeze(0)
                mem_ref_stg2_level_list.append(mem_ref_stg2)
            mem_ref_stg2_list.append(mem_ref_stg2_level_list)

        mem_ref_stg2_cat = [torch.cat(item, dim=1) for item in list(zip(*mem_ref_stg2_list))]



        mem_cur_stg2 = []
        for mem_ref_cat_level, mask_ref_cat_level, mem_cur_level, mask_cur_level in zip(mem_ref_stg1_cat, mask_ref_stg1_cat, mem_cur_stg1, mask_cur_stg1):
            mem_cur_stg2_level = self.s_decoder1(tgt=mem_cur_level, tgt_mask=None, memory=mem_ref_cat_level, memory_mask=None).squeeze(0)
            mem_cur_stg2.append(mem_cur_stg2_level)

        # mem_cur_stg2_cat = torch.cat(mem_cur_stg2, dim=1)

        # -------------------------------3rd stage------------------------------------
        mem_cur_stg3 = []
        for mem_cur, mem_ref_cat in zip(mem_cur_stg2, mem_ref_stg2_cat):
            mem_final_level = self.s_decoder2(tgt=mem_cur, memory=mem_ref_cat, memory_mask=None).squeeze(0)
            mem_cur_stg3.append(mem_final_level)

        mem_cur_stg3_cat = torch.cat(mem_cur_stg3, dim=1)

        # -------------------------------4th stage------------------------------------
        mem_cur_stg4 = []
        for mem_ref, mem_cur in zip(mem_cur_stg3, mem_cur_stg1):
            mem_level = self.our_decoder(tgt=mem_ref, memory=mem_cur, memory_mask=None).squeeze(0)
            mem_cur_stg4.append(mem_level)

        mem_cur_stg4_cat = torch.cat(mem_cur_stg4, dim=1)

        # -------------------------------5th stage------------------------------------
        # qln
        # mem_ref_stg4_list = [torch.cat(item, dim=1) for item in mem_ref_stg2_list]
        query_ref_list = []
        for mem_ref in mem_ref_stg1_cat4q:
            hs, reference_points, inter_references = self.pre_dec(mem_ref, query_embeds, dec_utils)
            query_ref = self.qln(hs[-1])
            query_ref_list.append(query_ref)

        # hs, reference_points, inter_references = self.d_dec(mem_ref_stg1_cat4q, query_embeds, dec_utils)
        # query_ref = self.qln(hs[-1])

        bs, _, _ = mem_cur_stg4_cat.shape
        query_embed = query_embeds.expand(bs, -1, -1)

        query_ref_cat = torch.cat(query_ref_list, dim=1)
        query_mix = torch.cat((query_embed, query_ref_cat), dim=1)

        # -------------------------------final stage------------------------------------
        hs, reference_points, inter_references = self.d_dec(mem_cur_stg4_cat, query_mix, dec_utils)

        return self.cal_loss(hs, reference_points, inter_references)



    def d_enc(self, srcs, masks, pos):
        src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = self.prepare_enc(srcs, masks, pos)

        memory = self.d_encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        return memory, (spatial_shapes, level_start_index, valid_ratios, mask_flatten)

    def d_dec(self, memory, query_embeds, dec_utils):
        spatial_shapes, level_start_index, valid_ratios, mask_flatten = dec_utils
        # print("spatial_shape", spatial_shapes)
        tgt, reference_points, query_embed = self.prepare_dec(memory, query_embeds)
        hs, inter_references = self.d_decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        return hs, reference_points, inter_references

    def pre_dec(self, memory, query_embeds, dec_utils):
        spatial_shapes, level_start_index, valid_ratios, mask_flatten = dec_utils
        # print("spatial_shape", spatial_shapes)
        tgt, reference_points, query_embed = self.prepare_dec1(memory, query_embeds)
        hs, inter_references = self.pre_decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        return hs, reference_points, inter_references

    def cal_loss_tc_like(self, hs, init_reference, inter_references):
        """
        TransCenter like without tracking branch
        :param hs:
        :param init_reference:
        :param inter_references:
        :return:
        """
        outputs_hms = []
        outputs_regs = []
        outputs_whs = []
        outputs_ct_offsets = []
        outputs_coords = []
        for layer_lvl in range(len(hs)):
            hs[layer_lvl] = self.ida_up[0](hs[layer_lvl], 0, len(hs[layer_lvl]))[-1]

            ct_offset = self.ct_offset(hs[layer_lvl])
            wh_head = self.wh(hs[layer_lvl])
            reg_head = self.reg(hs[layer_lvl])
            hm_head = self.hm(hs[layer_lvl])

            outputs_whs.append(wh_head)
            outputs_ct_offsets.append(ct_offset)
            outputs_regs.append(reg_head)
            outputs_hms.append(hm_head)
            outputs_coords.append(torch.cat([reg_head + ct_offset, wh_head], dim=1))
        out = {'hm': torch.stack(outputs_hms), 'boxes': torch.stack(outputs_coords),
               'wh': torch.stack(outputs_whs), 'reg': torch.stack(outputs_regs),
               'center_offset': torch.stack(outputs_ct_offsets)}

        return out

    def cal_loss(self, hs, init_reference, inter_references):
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out


    def get_feat(self, samples):
        features, pos = self.backbone(samples)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.cfg.MODEL.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.cfg.MODEL.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        return srcs, masks, pos

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def prepare_enc(self, srcs, masks, pos_embeds):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        return src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten

    def prepare_dec(self, memory, query_embed):
        bs, _, c = memory.shape

        if len(query_embed.shape) == 3:
            query_embed, tgt = torch.split(query_embed, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()

        return tgt, reference_points, query_embed

    def prepare_dec1(self, memory, query_embed):
        bs, _, c = memory.shape

        if len(query_embed.shape) == 3:
            query_embed, tgt = torch.split(query_embed, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points1(query_embed).sigmoid()

        return tgt, reference_points, query_embed

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def build_model(cfg):
    model_dict = {
        "PTSEFormer": PTSEFormer,
    }

    device = torch.device(cfg.TRAIN.device)
    model = model_dict[cfg.MODEL.name](cfg)
    # model = OURS_V4(cfg)
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.LOSS.cls_loss_coef, 'loss_bbox': cfg.LOSS.bbox_loss_coef}
    weight_dict['loss_giou'] = cfg.LOSS.giou_loss_coef
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(cfg.MODEL.num_classes, matcher, weight_dict, losses, focal_alpha=cfg.LOSS.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors


