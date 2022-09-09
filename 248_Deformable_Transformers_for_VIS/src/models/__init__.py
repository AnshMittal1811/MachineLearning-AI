# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from .backbone import build_backbone
from .matcher import build_matcher, build_inference_matcher
from .criterion import build_criterion
from .deformable_transformer import build_deformable_transformer
from .devis_ablation_transformer_wo_t_conn import build_devis_ablation_transformer
from .devis_transformer import build_devis_transformer
from .deformable_detr import DeformableDETR, DefDETRPostProcessor
from .deformable_segmentation import DeformableDETRSegm, DefDETRSegmPostProcess
from .devis_segmentation import DeVIS, DeVISPostProcessor
from .tracker import Tracker


def build_model(num_classes, device, cfg):
    backbone = build_backbone(cfg)
    matcher = build_matcher(cfg)
    criterion = build_criterion(matcher, num_classes, cfg)
    criterion.to(device)

    post_processor_kwargs = {
        'focal_loss': cfg.MODEL.LOSS.FOCAL_LOSS,
        'num_out': cfg.TEST.NUM_OUT,
        'use_top_k': cfg.TEST.USE_TOP_K,
    }

    deformable_detr_kwargs = {
        'backbone': backbone,
        'num_classes': num_classes - 1 if cfg.MODEL.LOSS.FOCAL_LOSS else num_classes,
        'num_queries': cfg.MODEL.NUM_QUERIES,
        'aux_loss': cfg.MODEL.LOSS.AUX_LOSS,
        'num_feature_levels': cfg.MODEL.NUM_FEATURE_LEVELS,
        'with_box_refine': cfg.MODEL.WITH_BBX_REFINE,
        'with_ref_point_refine': cfg.MODEL.WITH_REF_POINT_REFINE,
        'with_gradient': cfg.MODEL.BBX_GRADIENT_PROP
    }

    mask_kwargs = {
        'matcher': matcher,
        'use_deformable_conv': cfg.MODEL.MASK_HEAD.USE_MDC,
        'mask_head_used_features': cfg.MODEL.MASK_HEAD.USED_FEATURES,
        'att_maps_used_res': cfg.MODEL.MASK_HEAD.UPSAMPLING_RESOLUTIONS,
        'mask_aux_loss': cfg.MODEL.LOSS.MASK_AUX_LOSS,
    }

    if cfg.DATASETS.TYPE == 'vis':
        if cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.DISABLE_TEMPORAL_CONNECTIONS:
            transformer = build_devis_ablation_transformer(cfg)
        else:
            transformer = build_devis_transformer(cfg)
        deformable_detr_kwargs['transformer'] = transformer
        def_detr = DeformableDETR(**deformable_detr_kwargs)

        post_processor_kwargs['num_frames'] = cfg.MODEL.DEVIS.NUM_FRAMES
        postprocessors = DeVISPostProcessor(**post_processor_kwargs)

        mask_kwargs['defdetr'] = def_detr
        mask_kwargs['num_frames'] = cfg.MODEL.DEVIS.NUM_FRAMES
        mask_kwargs['post_processor'] = postprocessors
        mask_kwargs["add_3d_conv_head"] = cfg.MODEL.MASK_HEAD.DEVIS.CONV_HEAD_3D

        model = DeVIS(**mask_kwargs)

    else:
        transformer = build_deformable_transformer(cfg)
        deformable_detr_kwargs['transformer'] = transformer

        postprocessors = {
            'bbox': DefDETRPostProcessor(**post_processor_kwargs),
        }

        model = DeformableDETR(**deformable_detr_kwargs)

        if cfg.MODEL.MASK_ON:
            mask_kwargs['post_processor'] = postprocessors['bbox']
            mask_kwargs['defdetr'] = model
            postprocessors['segm'] = DefDETRSegmPostProcess()

            model = DeformableDETRSegm(**mask_kwargs)

    return model, criterion, postprocessors


def build_tracker(model, cfg):
    inference_matcher = build_inference_matcher(cfg)

    tracker_cfg = {
        "per_class_matching": cfg.TEST.CLIP_TRACKING.PER_CLASS_MATCHING,
        "track_min_detection_score": cfg.TEST.CLIP_TRACKING.MIN_FRAME_SCORE,
        "track_min_score": cfg.TEST.CLIP_TRACKING.MIN_TRACK_SCORE,
        "track_min_detections": cfg.TEST.CLIP_TRACKING.MIN_DETECTIONS,
        "final_class_policy": cfg.TEST.CLIP_TRACKING.FINAL_CLASS_POLICY,
        "final_score_policy": cfg.TEST.CLIP_TRACKING.FINAL_SCORE_POLICY,
    }

    tracker_visualization_cfg = {
        "out_viz_path": cfg.TEST.VIZ.OUT_VIZ_PATH,
        "save_clip_viz": cfg.TEST.VIZ.SAVE_CLIP_VIZ,
        "merge_tracks": cfg.TEST.VIZ.SAVE_MERGED_TRACKS,
    }

    tracker_kwargs = {
        'model': model,
        'hungarian_matcher': inference_matcher,
        'tracker_cfg': tracker_cfg,
        'visualization_cfg': tracker_visualization_cfg,
        'focal_loss': cfg.MODEL.LOSS.FOCAL_LOSS,
        'num_frames': cfg.MODEL.DEVIS.NUM_FRAMES,
        'overlap_window':  cfg.MODEL.DEVIS.NUM_FRAMES - cfg.TEST.CLIP_TRACKING.STRIDE,
        'use_top_k': cfg.TEST.USE_TOP_K,
        'num_workers': cfg.NUM_WORKERS,
    }

    return Tracker(**tracker_kwargs)
