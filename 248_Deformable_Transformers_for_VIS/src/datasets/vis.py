"""
YoutubeVIS data loader
"""
from pathlib import Path
import torch
import torch.utils.data
import torchvision.transforms as torch_T
from pycocotools.ytvos import YTVOS
import json
import os
from PIL import Image
import cv2
from . import vis_transforms as VisT


class VISTrainDataset:
    def __init__(self, ann_file: str, img_folder: str, transforms: VisT.VISTransformsApplier,
                 num_frames: int,
                 sample_each_frame: bool, focal_loss: bool):

        self.img_folder = img_folder
        self.sample_all = sample_each_frame
        self.focal_loss = focal_loss
        self.ann_file = ann_file
        self._transforms = transforms
        self.num_frames = num_frames
        self.prepare = VisT.ConvertCocoPolysToValuedMaskNumpy()
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.vid_ids = self.ytvos.getVidIds()
        self.vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)
        self.img_ids = []

        if not sample_each_frame:
            for idx, vid_info in enumerate(self.vid_infos):
                if vid_info["length"] < self.num_frames:
                    # Length video shorter than num_frames: We introduce padding as we do not want
                    # to ignore this clip
                    self.img_ids.append((idx, 0))
                    continue
                for frame_id in range(len(vid_info['filenames'])):
                    if frame_id + self.num_frames <= vid_info["length"]:
                        self.img_ids.append((idx, frame_id))
                    else:
                        break
        else:
            for idx, vid_info in enumerate(self.vid_infos):
                for frame_id in range(len(vid_info['filenames'])):
                    self.img_ids.append((idx, frame_id))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        vid, frame_id = self.img_ids[idx]
        vid_id = self.vid_infos[vid]['id']
        vid_len = self.vid_infos[vid]['length']
        raw_indices = list(range(0, - (vid_len - frame_id - 1), -1))

        img = []
        if len(raw_indices) >= self.num_frames:
            raw_indices = raw_indices[:self.num_frames]
        else:
            max_timestep = vid_len - frame_id - 1
            min_timestep = - frame_id
            list1 = list(range(-max_timestep, -min_timestep, 1))
            list2 = list(range(-min_timestep, -max_timestep, -1))
            while len(raw_indices) < self.num_frames:
                raw_indices.extend(list1 + list2)
            raw_indices = raw_indices[:self.num_frames]

        for j in range(self.num_frames):
            img_path = os.path.join(str(self.img_folder),
                                    self.vid_infos[vid]['file_names'][frame_id - raw_indices[j]])
            img.append(cv2.imread(img_path))

        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        target = self.ytvos.loadAnns(ann_ids)
        target = {'image_id': idx, 'video_id': vid, 'frame_id': frame_id, 'annotations': target}
        target = self.prepare(img[0], target, raw_indices, self.num_frames)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        target["num_trajectories"] = torch.tensor(target["labels"].shape[0] // self.num_frames)
        target["labels"] = target["labels"] - 1
        # Background is set to last logit
        num_cats = self.cat_ids[-1]
        for idx in range(target["labels"].shape[0]):
            if target["labels"][idx] == -1:
                target["labels"][idx] = num_cats

        if isinstance(img, list):
            img = torch.cat(img, dim=0)

        return img, target


class VideoClip(torch.utils.data.dataset.Dataset):

    def __init__(self, images_folder, video_id, video_clips, original_size, last_real_idx,
                 real_video_length, transform,
                 final_video_length, cat_names):
        self.video_id = video_id
        self.video_clips = video_clips
        self.last_real_idx = last_real_idx
        self.real_video_length = real_video_length
        self.images_folder = images_folder
        self.transform = transform
        self.original_size = original_size
        self.final_video_length = final_video_length
        self.cat_names = cat_names
        self.video_name = video_clips[0][0].split("/")[0]

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, item):
        video_clip = self.video_clips[item]
        clip_imgs_set = []
        for k in range(len(video_clip)):
            im = Image.open(os.path.join(self.images_folder, video_clip[k]))
            clip_imgs_set.append(self.transform(im).unsqueeze(0))
        img = torch.cat(clip_imgs_set, 0)
        return img


class VISValDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, ann_file, images_folder, transforms, max_clip_length, stride):
        self.ann_file = ann_file
        self.annotations = self._load_annotations()
        self.max_clip_length = max_clip_length
        self.overlap_window = max_clip_length - stride

        self.has_gt = "annotations" in self.annotations and self.annotations[
            "annotations"] is not None

        self.cat_names = {cat["id"]: cat["name"] for cat in self.annotations["categories"]}
        self.cat_names[0] = "Bkg"

        self._data = self.parse_video_into_clips(transforms, images_folder)

    def _load_annotations(self):
        with open(self.ann_file, 'r') as fh:
            annotations = json.load(fh)

        return annotations

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def get_total_num_frames(self):
        return sum([vid['length'] for vid in self.annotations['videos']])

    def parse_video_into_clips(self, transforms, images_folder):
        parsed_videos = []
        videos = self.annotations["videos"]

        for i in range(len(videos)):
            id_ = videos[i]['id']
            video_length = videos[i]['length']
            file_names = videos[i]['file_names']

            video_clips = []
            last_real_idx = 0
            real_video_length = None
            final_video_length = len(file_names)

            if video_length < self.max_clip_length:
                # TODO: Check this work properly
                video_to_read = []
                j = 1
                video_to_read.extend(file_names)
                while len(video_to_read) < self.max_clip_length:
                    if j % 2:
                        video_to_read.extend(file_names[::-1][1:])
                    else:
                        video_to_read.extend(file_names[1:])
                    j += 1
                video_clips.append(video_to_read[:self.max_clip_length])
                real_video_length = video_length

            elif video_length == self.max_clip_length:
                clip_names = file_names[:self.max_clip_length]
                video_clips.append(clip_names)

            else:
                first_clip = file_names[:self.max_clip_length]
                video_clips.append(first_clip)

                next_start_pos = self.max_clip_length - self.overlap_window
                next_end_pos = next_start_pos + self.max_clip_length

                while next_end_pos < video_length:
                    next_video_clip = file_names[next_start_pos:next_end_pos]
                    video_clips.append(next_video_clip)
                    next_start_pos = next_end_pos - self.overlap_window
                    next_end_pos = next_start_pos + self.max_clip_length

                last_clip_start_idx = len(file_names) - 1 - self.max_clip_length
                last_real_idx = next_start_pos - last_clip_start_idx - 1
                last_video_clip = file_names[-self.max_clip_length:]
                video_clips.append(last_video_clip)

            original_size = (videos[i]['height'], videos[i]['width'])
            parsed_videos.append(
                VideoClip(video_id=id_, video_clips=video_clips, last_real_idx=last_real_idx,
                          original_size=original_size, real_video_length=real_video_length,
                          transform=transforms, images_folder=images_folder,
                          final_video_length=final_video_length, cat_names=self.cat_names))

        return parsed_videos


def make_train_vis_transforms(out_scale, multi_scale_training, create_bbx_from_mask):
    scales_before_crop = [400, 500, 600]
    random_sized_crop = (384, 600)
    scales_before_crop = [int(out_scale * s) for s in scales_before_crop]
    random_sized_crop = tuple([int(out_scale * s) for s in random_sized_crop])

    if multi_scale_training:
        scales = [288, 320, 352, 392, 416, 448, 480, 512]
        max_size = 768
        if out_scale != 1.0:
            scales = [int(out_scale * s) for s in scales]
            max_size = int(max_size * out_scale)

        scales = (scales, max_size)
        scales_before_crop = (scales_before_crop, None)

        transforms = [
            VisT.VISHorizontalFlip(),
            VisT.VISPhotometricDistort(),
            VisT.VISRandomSelect(
                VisT.VISResize(scales),
                VisT.VISCompose([
                    VisT.VISResize(scales_before_crop),
                    VisT.VISRandomCrop(random_sized_crop),
                    VisT.VISResize(scales),
                ])
            ),
            VisT.VISToTensorWithPostProcessing(create_bbx_from_mask),
        ]

    else:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
        max_size = 800

        if out_scale != 1.0:
            scales = [int(out_scale * s) for s in scales]
            max_size = int(max_size * out_scale)

        out_shorter_edge = ([int(300 * out_scale)], int(540 * out_scale))
        scales = (scales, max_size)
        scales_before_crop = (scales_before_crop, None)

        transforms = [
            VisT.VISHorizontalFlip(),
            VisT.VISResize(scales),
            VisT.VISPhotometricDistort(),
            VisT.VISResize(scales_before_crop),
            VisT.VISRandomCrop(random_sized_crop),
            VisT.VISResize(out_shorter_edge),
            VisT.VISToTensorWithPostProcessing(create_bbx_from_mask),
        ]

    return VisT.VISTransformsApplier(transforms)


def make_val_vis_transforms(val_width, max_size):
    transform = torch_T.Compose([
        VisT.VISRandomClipResize([val_width], max_size=max_size),
        torch_T.ToTensor(),
        torch_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return transform


def build(image_set, cfg):
    split = cfg.get("DATASETS").get(f"{image_set}_DATASET")
    root = Path(cfg.DATASETS.DATA_PATH)
    assert root.exists(), f'provided Data path {root} does not exist'

    PATHS = {
        "yt_vis_train_19": ((root / "Youtube_VIS-2019/train/JPEGImages",
                             root / "Youtube_VIS-2019/train/" / 'train.json'), 40),
        "yt_vis_val_19": ((root / "Youtube_VIS-2019/valid/JPEGImages",
                           root / "Youtube_VIS-2019/valid/" / "valid.json"), 40),
        "yt_vis_train_21": ((root / "Youtube_VIS-2021/train/JPEGImages",
                             root / "Youtube_VIS-2021/train/" / 'instances.json'), 40),
        "yt_vis_train_21_wo_2975_2359": ((root / "Youtube_VIS-2021/train/JPEGImages",
                                          root / "Youtube_VIS-2021/train/" / 'instances_wo_2975_2359.json'), 40),
        "yt_vis_val_21": ((root / "Youtube_VIS-2021/valid/JPEGImages",
                           root / "Youtube_VIS-2021/valid/" / 'instances.json'), 40),
        "ovis_train": ((root / "OVIS/train/", root / "OVIS/" / "annotations_train.json"), 25),
        "ovis_val": ((root / "OVIS/valid/", root / "OVIS/" / "annotations_valid.json"), 25),

        "yt_vis_val_long": ((root / "Youtube_VIS-long/valid/JPEGImages",
                           root / "Youtube_VIS-long/valid/" / 'instances.json'), 40),

        # For debug purposes
        "mini_train": ((root / "Youtube_VIS/train/JPEGImages",
                        root / "Youtube_VIS/train/" / 'mini_train.json'), 40),
        "mini_val": ((root / "Youtube_VIS/valid/JPEGImages",
                      root / "Youtube_VIS/valid/" / 'mini_valid.json'), 40),

    }

    img_folder, ann_file = PATHS[split][0]
    num_classes = PATHS[split][1]

    assert os.path.isdir(img_folder), f"Provided VIS image folder path doesn't exist {img_folder}"
    assert os.path.isfile(ann_file), f"Provided VIS annotations file doesn't exist {ann_file}"

    if image_set == "TRAIN":
        transforms = make_train_vis_transforms(cfg.INPUT.SCALE_FACTOR_TRAIN,
                                               cfg.INPUT.DEVIS.MULTI_SCALE_TRAIN,
                                               cfg.INPUT.DEVIS.CREATE_BBX_FROM_MASK)
        dataset = VISTrainDataset(ann_file, img_folder, transforms,
                                  cfg.MODEL.DEVIS.NUM_FRAMES, cfg.INPUT.DEVIS.SAMPLE_EACH_FRAME,
                                  cfg.MODEL.LOSS.FOCAL_LOSS)
    else:
        transform = make_val_vis_transforms(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
        dataset = VISValDataset(ann_file, img_folder, transform, cfg.MODEL.DEVIS.NUM_FRAMES,
                                cfg.TEST.CLIP_TRACKING.STRIDE)

    return dataset, num_classes
