"""
Training script of DeVIS
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import argparse
import random
from contextlib import redirect_stdout
from pathlib import Path
import os
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

import src.util.misc as utils
from src.datasets import build_dataset
from src.models import build_model
from src.config import get_cfg_defaults
from main import sanity_check
from src.models.tracker import Tracker, Track, encode_mask
from src.util.att_maps_viz import visualize_clips_with_att_maps_per_reslvl, \
    visualize_clips_with_att_maps_merged_res_v2, create_masks


def get_args_parser():
    parser = argparse.ArgumentParser('DeVIS argument parser', add_help=False)
    parser.add_argument('--merge-resolution',
                        help="Allows converting all sampling location from "
                             "each resolution level to the same one.",
                        choices=[0, 1, 2, 3],
                        type=int,
                        default=None)

    parser.add_argument('--layer',
                        help="Allows selecting the layer in which visualizing the attention maps",
                        choices=[0, 1, 2, 3, 4, 5],
                        type=int,
                        default=5)

    parser.add_argument('--used-resolution',
                        help="If merge_resolution=None, allows selecting the resolution to save attention maps",
                        choices=[0, 1, 2, 3], default=1)

    parser.add_argument('--config-file', help="Run test only")
    parser.add_argument('--eval-only', action='store_true', help="Run test only")

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument(
        "opts",
        help="""
    Modify config options at the end of the command. For Yacs configs, use
    space-separated "PATH.KEY VALUE" pairs".
            """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


# TODO: Implement Attention maps visualization when
#  MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_CONNECT_ALL_FRAME = False
class TrackerAttMaps(Tracker):

    def __init__(self, model: torch.nn.Module, hungarian_matcher, tracker_cfg: dict,
                 att_maps_cfg: dict, visualization_cfg: dict, focal_loss: bool, num_frames: int,
                 overlap_window: int,
                 use_top_k: bool, num_workers: int):

        super().__init__(model, hungarian_matcher, tracker_cfg, visualization_cfg, focal_loss,
                         num_frames, overlap_window, use_top_k, num_workers)
        self.att_maps_cfg = utils.nested_dict_to_namespace(att_maps_cfg)

    def process_masks(self, start_idx, idx, tgt_size, masks):
        processed_masks = []
        num_masks = masks.shape[0]
        for t in range(num_masks):
            mask = masks[t]
            mask = F.interpolate(mask[None, None], tgt_size, mode="bilinear",
                                 align_corners=False).detach().sigmoid()[0, 0]
            processed_masks.append(encode_mask(mask))

        return processed_masks

    def parse_att_maps_to_tracks(self, tracks, topk_idxs, merge_resolution, spatial_shapes,
                                 init_ref_point, inter_ref_points, sampling_locations,
                                 temporal_sampling_locations, attn_weights, temporal_attn_weights):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = self.model.module
        else:
            model_without_ddp = self.model
        t_window = model_without_ddp.def_detr.transformer.decoder.layers[-1].cross_attn.t_window
        n_levels = model_without_ddp.def_detr.transformer.decoder.layers[-1].cross_attn.n_levels
        embd_per_frame = sampling_locations[0].shape[1]

        sampling_offsets = torch.cat(sampling_locations, dim=0)
        coordinates_lvl_res_factor = torch.flip(spatial_shapes, dims=(1,))
        if merge_resolution is not None:
            coordinates_lvl_res_factor = coordinates_lvl_res_factor[merge_resolution].repeat(
                spatial_shapes.shape[0], 1)

        sampling_locations = sampling_offsets[:, topk_idxs] * coordinates_lvl_res_factor[None, None,
                                                              None, :, None]

        temporal_sampling_offsets = torch.cat(temporal_sampling_locations, dim=0).unflatten(3, [
            t_window, n_levels])
        temporal_sampling_locations = temporal_sampling_offsets[:,
                                      topk_idxs] * coordinates_lvl_res_factor[None, None, None,
                                                   None, :, None]

        attn_weights = attn_weights[:, topk_idxs]
        temporal_attn_weights = temporal_attn_weights.unflatten(3, [t_window, n_levels])[:,
                                topk_idxs]

        if self.att_maps_cfg.layer == 0:
            ref_points = init_ref_point[0].sigmoid()
        else:
            ref_points = inter_ref_points[self.att_maps_cfg.layer - 1, 0]

        ref_points = ref_points.reshape(
            [model_without_ddp.num_frames, embd_per_frame, ref_points.shape[-1]])[:, topk_idxs]

        for i, track in enumerate(tracks):
            # TODO: This round should be replaced by proper interpolation of corresponding pixels
            track.curr_position = torch.round(sampling_locations[:, i]).type(torch.long)
            track.curr_att_weights = attn_weights[:, i]

            track.temporal_positions = torch.round(temporal_sampling_locations[:, i]).type(
                torch.long)
            track.temporal_att_weights = temporal_attn_weights[:, i]
            track.spatial_shapes = spatial_shapes
            track.ref_point = ref_points[:, i]

    def __call__(self, video, device, all_times):
        sampler_val = torch.utils.data.SequentialSampler(video)
        video_loader = DataLoader(video, 1, sampler=sampler_val, num_workers=self.num_workers)
        real_video_length = video.real_video_length
        clip_length = self.num_frames if real_video_length is None or real_video_length >= self.num_frames else real_video_length
        cat_names = video.cat_names
        video_info = {
            "tgt_size": video.original_size,
            "clip_length": clip_length
        }

        # use lists to store the outputs via up-values
        init_ref_point, inter_ref_points, raw_deformable_attention = [], [], []
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = self.model.module
        else:
            model_without_ddp = self.model

        hooks = [
            model_without_ddp.def_detr.transformer.reference_points.register_forward_hook(
                lambda self, input, output: init_ref_point.append(output)
            ),
            model_without_ddp.def_detr.transformer.decoder.layers[
                self.att_maps_cfg.layer].cross_attn.register_forward_hook(
                lambda self, input, output: raw_deformable_attention.append(output[1:])
            ),
            model_without_ddp.def_detr.transformer.decoder.register_forward_hook(
                lambda self, input, output: inter_ref_points.append(output[1])
            ),
        ]

        for idx, video_clip in enumerate(video_loader):
            init_ref_point, inter_ref_points, raw_deformable_attention = [], [], []
            clip_tracks_category_dict = {}
            video_clip = video_clip.to(device)

            results = self.model(video_clip.squeeze(0), video_info)
            init_ref_point = init_ref_point[0]
            inter_ref_points = inter_ref_points[0]
            sampling_offsets, temporal_sampling_offsets, attn_weights, temporal_attn_weights = \
            raw_deformable_attention[0]

            pred_scores, pred_classes, pred_boxes, pred_masks, pred_center_points = results["scores"], \
                                                                                    results["labels"], results["boxes"], \
                                                                                    results["masks"], results["center_points"]
            detected_instances = pred_scores.shape[1]

            start_idx = 0 if idx != len(video_loader) - 1 else video.last_real_idx
            clip_tracks = [Track(track_id, clip_length, start_idx) for track_id in
                           range(detected_instances)]

            processed_masks_dict = {}

            for i, track in enumerate(clip_tracks):
                mask_id = results['inverse_idxs'][i].item()
                if mask_id not in processed_masks_dict.keys():
                    processed_masks_dict[mask_id] = self.process_masks(start_idx, idx,
                                                                       video.original_size,
                                                                       pred_masks[:, mask_id])

                cat_track = pred_classes[0, i].item()
                if cat_track not in clip_tracks_category_dict:
                    clip_tracks_category_dict[cat_track] = []

                clip_tracks_category_dict[cat_track].append(i)
                track.update(pred_scores[:, i], pred_classes[:, i], pred_boxes[:, i],
                             processed_masks_dict[mask_id], pred_center_points[:, i], mask_id)

            self.parse_att_maps_to_tracks(clip_tracks, results["top_k_idxs"],
                                          self.att_maps_cfg.merge_resolution,
                                          results['spatial_shapes'], init_ref_point,
                                          inter_ref_points, sampling_offsets,
                                          temporal_sampling_offsets, attn_weights,
                                          temporal_attn_weights)

            if self.tracker_cfg.track_min_detection_score != 0:
                for track in clip_tracks:
                    track.filter_frame_detections(self.tracker_cfg.track_min_detection_score)

            keep = np.array([track.valid(min_detections=1) for track in clip_tracks])
            clips_to_show = [track for i, track in enumerate(clip_tracks) if keep[i]]

            if self.tracker_cfg.track_min_score != 0:
                keep = [track.mean_score() > self.tracker_cfg.track_min_score for track in
                        clips_to_show]
                clips_to_show = [track for i, track in enumerate(clips_to_show) if keep[i]]

            for track in clips_to_show:
                track.encode_all_masks()
                track.process_centroid(video.original_size)

            if self.att_maps_cfg.merge_resolution is None:
                visualize_clips_with_att_maps_per_reslvl(idx, video.images_folder,
                                                         video.video_clips[idx][:clip_length],
                                                         clips_to_show,
                                                         self.att_maps_cfg.layer,
                                                         self.att_maps_cfg.used_resolution,
                                                         out_path=self.visualization_cfg.out_viz_path,
                                                         class_name=cat_names)
            else:
                visualize_clips_with_att_maps_merged_res_v2(idx, video.images_folder,
                                                            video.video_clips[idx][:clip_length],
                                                            clips_to_show,
                                                            self.att_maps_cfg.layer,
                                                            self.att_maps_cfg.merge_resolution,
                                                            out_path=self.visualization_cfg.out_viz_path,
                                                            class_name=cat_names)

        for hook in hooks:
            hook.remove()


@torch.no_grad()
def run_demo(args, cfg):
    sanity_check(cfg)

    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = cfg.SEED + utils.get_rank()

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_val, num_classes = build_dataset(image_set="VAL", cfg=cfg)
    model, criterion, postprocessors = build_model(num_classes, device, cfg)
    model.to(device)
    model.eval()
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    tracker = TrackerAttMaps(
        model=model,
        hungarian_matcher=None,
        tracker_cfg={
            "per_class_matching": cfg.TEST.CLIP_TRACKING.PER_CLASS_MATCHING,
            "track_min_detection_score": cfg.TEST.CLIP_TRACKING.MIN_FRAME_SCORE,
            "track_min_score": cfg.TEST.CLIP_TRACKING.MIN_TRACK_SCORE,
            "track_min_detections": cfg.TEST.CLIP_TRACKING.MIN_DETECTIONS,
            "final_class_policy": cfg.TEST.CLIP_TRACKING.FINAL_CLASS_POLICY,
            "final_score_policy": cfg.TEST.CLIP_TRACKING.FINAL_SCORE_POLICY,
        },
        num_workers=cfg.NUM_WORKERS,
        use_top_k=cfg.TEST.USE_TOP_K,
        overlap_window=cfg.MODEL.DEVIS.NUM_FRAMES - cfg.TEST.CLIP_TRACKING.STRIDE,
        num_frames=cfg.MODEL.DEVIS.NUM_FRAMES,
        focal_loss=cfg.MODEL.LOSS.FOCAL_LOSS,
        visualization_cfg={
            "out_viz_path": cfg.TEST.VIZ.OUT_VIZ_PATH,
            "save_clip_viz": cfg.TEST.VIZ.SAVE_CLIP_VIZ,
            "merge_tracks": cfg.TEST.VIZ.SAVE_MERGED_TRACKS,
        },
        att_maps_cfg={
            "merge_resolution": args.merge_resolution,
            "layer": args.layer,
            "used_resolution": args.used_resolution,
        }
    )

    n_total_params = sum(p.numel() for p in model.parameters())
    print(f'Total num params: {n_total_params}')

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, cfg.SOLVER.BATCH_SIZE, sampler=sampler_val,
                                 collate_fn=utils.val_collate if cfg.DATASETS.TYPE == 'vis' else utils.collate_fn,
                                 num_workers=cfg.NUM_WORKERS)


    resume_state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location=device)['model']
    model_without_ddp.load_state_dict(resume_state_dict, strict=True)
    selected_videos = False
    if cfg.TEST.VIZ.VIDEO_NAMES:
        selected_videos = cfg.TEST.VIZ.VIDEO_NAMES.split(",")
    for idx, video in tqdm.tqdm(enumerate(data_loader_val)):
        if selected_videos and video.video_clips[0][0].split("/")[0] not in selected_videos:
            continue
        tracker(video, device, [])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DeVIS attention maps demo script',
                                     parents=[get_args_parser()])
    args_ = parser.parse_args()

    cfg_ = get_cfg_defaults()
    cfg_.merge_from_file(args_.config_file)
    cfg_.merge_from_list(args_.opts)
    cfg_.freeze()
    if cfg_.OUTPUT_DIR:
        Path(cfg_.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(cfg_.OUTPUT_DIR, 'config.yaml'), 'w') as yaml_file:
            with redirect_stdout(yaml_file):
                print(cfg_.dump())

    run_demo(args_, cfg_)
