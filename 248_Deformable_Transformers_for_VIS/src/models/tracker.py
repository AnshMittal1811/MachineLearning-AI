import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pycocotools.mask as mask_util
import copy
import time

from ..util.misc import nested_dict_to_namespace
from ..util.viz_utils import visualize_tracks_independently, visualize_clips_after_processing, visualize_results_merged


class Track:
    def __init__(self, track_id, track_length, start_idx=0):
        self._id = track_id
        self.length = track_length
        self.start_idx = start_idx
        self.attrs = ["scores", "masks", "categories", "boxes", "centroid_points"]
        self.scores = [None for _ in range(track_length)]
        self.masks = [None for _ in range(track_length)]
        self.categories = [None for _ in range(track_length)]
        self.boxes = [None for _ in range(track_length)]
        self.centroid_points = [None for _ in range(track_length)]
        self.valid_frames = [True for _ in range(track_length)]
        self.mask_id = None
        self.last_t = 0
        self.matching_ids_record = []

    def __len__(self):
        return self.length

    def get_id(self):
        return self._id

    def valid(self, min_detections=1):
        return sum(self.valid_frames) >= min_detections

    def update(self, scores, categories, boxes, masks, centroid_points, mask_id):
        self.scores = scores.tolist()
        self.categories = categories.tolist()
        self.boxes = boxes.tolist()
        self.centroid_points = centroid_points.tolist()
        self.mask_id = mask_id
        self.masks = masks

    def init_video_track(self, num_t, track):
        for attr in self.attrs:
            getattr(self, attr)[:num_t] = getattr(track, attr)
        self.mask_id = track.mask_id

    def mean_score(self):
        scores = [score for valid, score in zip(self.valid_frames, self.scores) if valid]
        if not scores:
            return 0
        else:
            return np.mean(scores)

    def median_score(self):
        scores = [score for valid, score in zip(self.valid_frames, self.scores) if valid]
        if not scores:
            return 0
        else:
            return np.median(scores)

    def get_last_t_result(self, t, attr):
        return getattr(self, attr)[self.last_t + t]

    def get_last_results(self, t_window, attr):
        return getattr(self, attr)[self.last_t - t_window: self.last_t]

    def get_mask_id(self):
        return self.mask_id

    def get_first_t_result(self, t, attr):
        return getattr(self, attr)[(self.start_idx + t)]

    def get_first_results(self, t_window, attr):
        return getattr(self, attr)[self.start_idx: self.start_idx + t_window]

    def get_results_to_append(self, t, attr):
        return getattr(self, attr)[(self.start_idx + t):]

    def get_results_to_start(self, attr):
        return getattr(self, attr)

    def add_video_track(self, t_window, track):
        start_idx = self.last_t - t_window - track.start_idx
        for i in range(start_idx):
            self.valid_frames[i] = False

        for attr in self.attrs:
            results = getattr(track, attr)
            getattr(self, attr)[start_idx: start_idx + len(results)] = results

        self.mask_id = track.mask_id
        self.cat_for_matcher = track.categories[0]

    def kill_track(self):
        for i in range(self.last_t, len(self.valid_frames)):
            self.valid_frames[i] = False
        self.encode_all_masks()

    def append_track(self, track, t_window: int):
        # Overwrite all current detections for the overlap frames
        self.matching_ids_record.append((self._id, track.get_id()))
        overlap_positions = range(self.last_t - t_window - track.start_idx, self.last_t)
        # We pick detections with the highest score on the overlap frames
        for other_pos, self_pos in enumerate(overlap_positions):
            other_score = track.scores[other_pos] if track.scores[other_pos] is not None else 0
            self_score = self.scores[self_pos] if self.scores[self_pos] is not None else 0
            if other_score > self_score:
                for attr in self.attrs:
                    if attr == "masks":
                        other_mask = track.masks[other_pos]
                        self.masks[self_pos] = other_mask
                    else:
                        getattr(self, attr)[self_pos] = getattr(track, attr)[other_pos]

        # Fill non-overlapping frames
        for attr in self.attrs:
            results = track.get_results_to_append(t_window, attr)
            getattr(self, attr)[self.last_t:self.last_t + len(results)] = results

        self.mask_id = track.mask_id

    def update_stride(self, stride):
        self.last_t += stride

    def update_stride_and_encode_masks(self, stride, overlap_window):
        # Check if mask can be encoded and save
        for idx in range(self.last_t - overlap_window, self.last_t - overlap_window + stride):
            if 0 <= idx < len(self.masks):
                if self.masks[idx] is None:
                    self.valid_frames[idx] = False
                elif not isinstance(self.masks[idx], dict):
                    self.masks[idx] = encode_mask(self.masks[idx])

        self.last_t += stride

    def filter_frame_detections(self, min_detection_score):
        # Eliminates individual detections with score < min_detection_score
        for idx, score in enumerate(self.scores):
            if score is None:
                assert not self.valid_frames[idx]
            else:
                if score < min_detection_score:
                    self.valid_frames[idx] = False

    def compute_final_score(self, policy):
        if policy == "mean":
            return self.mean_score()
        elif policy == "median":
            return self.median_score()
        else:
            raise ValueError("Score policy not implemented: Available: mean, median")

    def compute_final_category(self, policy):
        if policy == "most_common":
            category_ids = np.array([cat for valid, cat in zip(self.valid_frames, self.categories) if valid])
            return np.argmax(np.bincount(category_ids))

        elif policy == "score_weighting":
            score_dict = {}
            for valid, score, cat in zip(self.valid_frames, self.scores, self.categories):
                if not valid:
                    continue
                if cat not in score_dict:
                    score_dict[cat] = 0
                score_dict[cat] += score
            values, idx_dict = [], {}
            for idx, (key, value) in enumerate(score_dict.items()):
                values.append(value)
                idx_dict[idx] = key
            cat_idx = np.array(values).argmax()
            return idx_dict[cat_idx]

        else:
            raise ValueError("Category policy not implemented: Available: most_common, score_weighting")

    def get_formatted_result(self, video_id, category_policy, score_policy):
        final_score = self.compute_final_score(score_policy)
        final_category_id = self.compute_final_category(category_policy)
        final_masks = []
        for valid, mask in zip(self.valid_frames, self.masks):
            if valid:
                final_masks.append(mask)
            else:
                final_masks.append(None)

        track = {'video_id': int(video_id),
                 'score': float(final_score),
                 'category_id': int(final_category_id),
                 'segmentations': final_masks}

        return track

    def encode_all_masks(self):
        for t in range(self.length):
            if self.masks[t] is not None and not isinstance(self.masks[t], dict):
                self.masks[t] = encode_mask(self.masks[t])

    def process_centroid(self, tgt_size):
        # For visualization purposes only
        img_h, img_w = torch.tensor([tgt_size]).unbind(-1)
        scale_fct = torch.stack([img_w, img_h], dim=1)
        new_centroids = []
        for centroid_p in self.centroid_points:
            if centroid_p is not None:
                centroid_points = torch.tensor(centroid_p)[None] * scale_fct
                centroid_points[:, 0].clamp_(0, img_w.item())
                centroid_points[:, 1].clamp_(0, img_h.item())
                new_centroids.append(centroid_points.tolist()[0])
            else:
                new_centroids.append(None)

        self.centroid_points = new_centroids


def encode_mask(mask):
    mask = mask.cpu().numpy() > 0.5
    rle_mask = mask_util.encode(np.array(mask, order='F'))
    rle_mask["counts"] = rle_mask["counts"].decode("utf-8")
    return rle_mask


class Tracker:
    def __init__(self, model: torch.nn.Module, hungarian_matcher, tracker_cfg: dict, visualization_cfg: dict, focal_loss: bool,
                 num_frames: int, overlap_window: int, use_top_k: bool, num_workers: int):
        self.model = model
        self.hungarian_matcher = hungarian_matcher
        self.tracker_cfg = nested_dict_to_namespace(tracker_cfg)
        self.visualization_cfg = nested_dict_to_namespace(visualization_cfg)
        self.focal_loss = focal_loss
        self.num_frames = num_frames
        self.overlap_window = overlap_window
        self.use_top_k = use_top_k
        self.num_workers = num_workers

    def process_masks(self, start_idx, idx, tgt_size, masks):
        processed_masks = []
        num_masks = masks.shape[0]
        for t in range(num_masks):
            mask = masks[t]
            mask = F.interpolate(mask[None, None], tgt_size, mode="bilinear", align_corners=False).detach().sigmoid()[0, 0]
            if self.hungarian_matcher.use_binary_mask_iou:
                processed_masks.append(encode_mask(mask))
            else:
                # We encode masks that will never be used for clip stitching, so they do not waste gpu memory
                if idx == 0:
                    if t < num_masks - self.overlap_window:
                        processed_masks.append(encode_mask(mask))
                    else:
                        processed_masks.append(mask)
                else:
                    if self.overlap_window + start_idx <= t < num_masks - self.overlap_window or t < start_idx:
                        processed_masks.append(encode_mask(mask))
                    else:
                        processed_masks.append(mask)

        return processed_masks

    def visualize_clip_results(self, video, clip_tracks, idx, cat_names, clip_length):
        clips_to_show = copy.deepcopy(clip_tracks)

        if self.tracker_cfg.track_min_detection_score != 0:
            for track in clips_to_show:
                track.filter_frame_detections(self.tracker_cfg.track_min_detection_score)
        keep = np.array([track.valid(min_detections=1) for track in clips_to_show])
        clips_to_show = [track for i, track in enumerate(clips_to_show) if keep[i]]

        if self.tracker_cfg.track_min_score != 0:
            keep = [track.mean_score() > self.tracker_cfg.track_min_score for track in clips_to_show]
            clips_to_show = [track for i, track in enumerate(clips_to_show) if keep[i]]
        for track in clips_to_show:
            track.encode_all_masks()
            track.process_centroid(video.original_size)

        visualize_clips_after_processing(idx, video.images_folder, video.video_clips[idx][:clip_length], clips_to_show, out_path=self.visualization_cfg.out_viz_path,
                                         class_name=cat_names)

    def add_new_clips(self, video, video_track_ids_counter, cat_clip_tracks_ids, video_tracks, video_tracks_category_dict, video_timestep, clip_tracks, cat_to_match):

        for track_id_to_add in cat_clip_tracks_ids:
            video_track_ids_counter += 1
            next_id = video_track_ids_counter
            track_to_add = Track(next_id, video.final_video_length, 0)
            track_to_add.update_stride(video_timestep)
            track_to_add.add_video_track(self.overlap_window, clip_tracks[track_id_to_add])
            video_tracks.append(track_to_add)
            if cat_to_match not in video_tracks_category_dict:
                video_tracks_category_dict[cat_to_match] = []
            video_tracks_category_dict[cat_to_match].append(next_id)

        return video_track_ids_counter, video_tracks, video_tracks_category_dict, video_timestep

    @torch.no_grad()
    def __call__(self, video, device, all_times):
        sampler_val = torch.utils.data.SequentialSampler(video)
        video_loader = DataLoader(video, 1, sampler=sampler_val, num_workers=self.num_workers)
        real_video_length = video.real_video_length
        clip_length = self.num_frames if real_video_length is None or real_video_length >= self.num_frames else real_video_length
        cat_names = video.cat_names
        video_tracks = None
        video_info = {
            "tgt_size": video.original_size,
            "clip_length": clip_length
        }

        times = []

        # Used for per-class matching only
        video_track_ids_counter = 0
        video_tracks_category_dict = {}
        video_timestep = 0

        for idx, video_clip in enumerate(video_loader):
            clip_tracks_category_dict = {}
            video_clip = video_clip.to(device)

            time1 = time.time()
            results = self.model(video_clip.squeeze(0), video_info)
            time_inference = time.time() - time1
            times.append(time_inference)

            pred_scores, pred_classes, pred_boxes, pred_masks, pred_center_points = results["scores"], results["labels"], results["boxes"], results["masks"], results["center_points"]
            detected_instances = pred_scores.shape[1]

            start_idx = 0 if idx != len(video_loader) - 1 else video.last_real_idx
            clip_tracks = [Track(track_id, clip_length, start_idx) for track_id in range(detected_instances)]

            processed_masks_dict = {}

            for i, track in enumerate(clip_tracks):
                mask_id = results['inverse_idxs'][i].item()
                if mask_id not in processed_masks_dict.keys():
                    processed_masks_dict[mask_id] = self.process_masks(start_idx, idx, video.original_size, pred_masks[:, mask_id])

                cat_track = pred_classes[0, i].item()
                if cat_track not in clip_tracks_category_dict:
                    clip_tracks_category_dict[cat_track] = []

                clip_tracks_category_dict[cat_track].append(i)
                track.update(pred_scores[:, i], pred_classes[:, i], pred_boxes[:, i], processed_masks_dict[mask_id], pred_center_points[:, i], mask_id)

            time1 = time.time()

            if self.visualization_cfg.save_clip_viz and self.visualization_cfg.out_viz_path:
                self.visualize_clip_results(video, clip_tracks, idx, cat_names, clip_length)

            if video_tracks is None:
                # Initialize videos
                video_track_ids_counter += detected_instances - 1
                video_tracks = [Track(track_id, video.final_video_length, start_idx) for track_id in range(detected_instances)]
                for new_track in clip_tracks:
                    cat_track = new_track.categories[0]
                    if cat_track not in video_tracks_category_dict:
                        video_tracks_category_dict[cat_track] = []
                    video_tracks_category_dict[cat_track].append(new_track.get_id())
                    video_tracks[new_track.get_id()].init_video_track(clip_length, new_track)

                video_timestep += clip_length
                for track in video_tracks:
                    track.update_stride(clip_length)

            else:
                # Perform stitching
                if not self.tracker_cfg.per_class_matching:
                    matched_tracks1_ids, matched_tracks2_ids = self.hungarian_matcher(video_tracks, clip_tracks)

                    for track_pos1, track_pos2 in zip(matched_tracks1_ids, matched_tracks2_ids):
                        video_tracks[track_pos1].append_track(clip_tracks[track_pos2], self.overlap_window)

                    for track in video_tracks:
                        track.update_stride_and_encode_masks(clip_length - self.overlap_window, self.overlap_window)

                else:
                    existing_cats = set(list(video_tracks_category_dict.keys()) + list(clip_tracks_category_dict.keys()))
                    for cat_to_match in existing_cats:
                        cat_video_track_ids = video_tracks_category_dict[cat_to_match] if cat_to_match in video_tracks_category_dict else []
                        cat_clip_tracks_ids = clip_tracks_category_dict[cat_to_match] if cat_to_match in clip_tracks_category_dict else []

                        # Case 1: This category doesnt exist in clip 2 (future)
                        if not cat_clip_tracks_ids:
                            # Remove all tracks from that category
                            for vid_id_to_remove in cat_video_track_ids:
                                video_tracks[vid_id_to_remove].kill_track()
                            del video_tracks_category_dict[cat_to_match]

                        # Case 2: This category doesn't exist in clip 1 (past)
                        elif not cat_video_track_ids:
                            # Create new Video track (full length) and add corresponding clip data into it
                            video_track_ids_counter, video_tracks, video_tracks_category_dict, video_timestep = self.add_new_clips(video, video_track_ids_counter,
                                                                                                                                   cat_clip_tracks_ids, video_tracks,
                                                                                                                                   video_tracks_category_dict, video_timestep,
                                                                                                                                   clip_tracks,
                                                                                                                                   cat_to_match)

                        # Case 3: Exist on both, perform matching of these tracks
                        else:
                            video_tracks_to_match = [video_tracks[id_] for id_ in cat_video_track_ids]
                            clip_tracks_to_match = [clip_tracks[id_] for id_ in cat_clip_tracks_ids]

                            matched_tracks1_index, matched_tracks2_index = self.hungarian_matcher(video_tracks_to_match, clip_tracks_to_match)

                            matched_video_track_ids = [cat_video_track_ids[idx_] for idx_ in matched_tracks1_index]
                            matched_clip_track_ids = [cat_clip_tracks_ids[idx_] for idx_ in matched_tracks2_index]

                            for track_pos1, track_pos2 in zip(matched_video_track_ids, matched_clip_track_ids):
                                video_tracks[track_pos1].append_track(clip_tracks[track_pos2], self.overlap_window)

                            # Kill unmatched tracks
                            video_ids_to_remove = list(set(cat_video_track_ids) - set(matched_video_track_ids))
                            for video_id_to_remove in video_ids_to_remove:
                                video_tracks[video_id_to_remove].kill_track()
                                video_tracks_category_dict[cat_to_match].remove(video_id_to_remove)

                            # New tracks to add
                            clip_ids_to_add = list(set(cat_clip_tracks_ids) - set(matched_clip_track_ids))

                            video_track_ids_counter, video_tracks, video_tracks_category_dict, video_timestep = self.add_new_clips(video, video_track_ids_counter, clip_ids_to_add,
                                                                                                                                   video_tracks, video_tracks_category_dict,
                                                                                                                                   video_timestep, clip_tracks, cat_to_match)

                    stride = clip_length - self.overlap_window
                    video_timestep += stride
                    for track in video_tracks:
                        track.update_stride_and_encode_masks(stride, self.overlap_window)

            time_tracking = time.time() - time1
            times.append(time_tracking)

        all_times.append(sum(times))

        if self.tracker_cfg.track_min_detection_score != 0:
            for track in video_tracks:
                track.filter_frame_detections(self.tracker_cfg.track_min_detection_score)

        keep = np.array([track.valid(min_detections=self.tracker_cfg.track_min_detections) for track in video_tracks])
        video_tracks = [track for i, track in enumerate(video_tracks) if keep[i]]

        if self.tracker_cfg.track_min_score != 0:
            keep = [track.compute_final_score(self.tracker_cfg.final_score_policy) > self.tracker_cfg.track_min_score for track in video_tracks]
            video_tracks = [track for i, track in enumerate(video_tracks) if keep[i]]

        # Sanity check all masks are encoded
        if not self.hungarian_matcher.use_binary_mask_iou:
            for track in video_tracks:
                track.encode_all_masks()

        if self.visualization_cfg.out_viz_path:
            for track in video_tracks:
                track.process_centroid(video.original_size)
            if self.visualization_cfg.merge_tracks:
                visualize_results_merged(video.images_folder, video.video_clips, video_tracks, self.tracker_cfg.final_class_policy,
                                         self.tracker_cfg.final_score_policy, out_path=self.visualization_cfg.out_viz_path, class_name=cat_names)
            else:
                visualize_tracks_independently(video.images_folder, video.video_clips, video_tracks, self.tracker_cfg.final_class_policy,
                                               self.tracker_cfg.final_score_policy, out_path=self.visualization_cfg.out_viz_path, class_name=cat_names)

        final_tracks_result = [track.get_formatted_result(video.video_id, self.tracker_cfg.final_class_policy, self.tracker_cfg.final_score_policy) for track in video_tracks]

        return final_tracks_result, all_times
