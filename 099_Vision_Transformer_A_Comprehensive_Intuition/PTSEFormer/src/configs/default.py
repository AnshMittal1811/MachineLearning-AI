import os
import numpy as np
from yacs.config import CfgNode as CN


_C = CN()

_C.TRAIN = CN()

_C.TRAIN.lr = 2e-4
_C.TRAIN.lr_backbone_names = ["backbone.0"]
_C.TRAIN.lr_backbone = 2e-5
_C.TRAIN.lr_linear_proj_names = ['reference_points', 'sampling_offsets']
_C.TRAIN.lr_linear_proj_mult = 0.1
_C.TRAIN.batch_size = 2
_C.TRAIN.weight_decay = 1e-4
_C.TRAIN.epochs = 50
_C.TRAIN.lr_drop = 40
_C.TRAIN.lr_drop_epochs = None
_C.TRAIN.clip_max_norm = 0.1
_C.TRAIN.sgd = True
_C.TRAIN.output_dir = "../results_DETR"
_C.TRAIN.device = "cuda"
_C.TRAIN.seed = 42
_C.TRAIN.resume = ""
_C.TRAIN.resume_default = True
_C.TRAIN.pretrained = "../pretrained/resnet50-19c8e357.pth"
_C.TRAIN.start_epoch = 0
_C.TRAIN.eval = True
_C.TRAIN.num_workers = 0
_C.TRAIN.cache_mode = False

_C.TEST = CN()
_C.TEST.batch_size = 1
_C.TEST.test_with_one_img = False
_C.TEST.test_ref_nums = 2
_C.TEST.test_max_offset = 12
_C.TEST.test_min_offset = -12

_C.MODEL = CN()

_C.MODEL.name = "DeformDETR" # "TCDet", "DeformDETR"
_C.MODEL.frozen_weights = None
_C.MODEL.backbone = "resnet50"
_C.MODEL.dilation = True
_C.MODEL.position_embedding = "sine"  # choices=('sine', 'learned')
_C.MODEL.position_embedding_scale = 2 * np.pi
_C.MODEL.num_feature_levels = 4
_C.MODEL.enc_layers = 6
_C.MODEL.dec_layers = 6
_C.MODEL.dim_feedforward = 1024
_C.MODEL.hidden_dim = 256
_C.MODEL.dropout = 0.1
_C.MODEL.nheads = 8
_C.MODEL.num_quries = 300
_C.MODEL.dec_n_points = 4
_C.MODEL.enc_n_points = 4
_C.MODEL.masks = False
_C.MODEL.num_classes = 91
_C.MODEL.num_queries = 300

# Loss
_C.LOSS = CN()

_C.LOSS.no_aux_loss = True

# these are for Deformable DETR
_C.LOSS.set_cost_class = 2
_C.LOSS.set_cost_bbox = 5
_C.LOSS.set_cost_giou = 2

_C.LOSS.mask_loss_coef = 1
_C.LOSS.dice_loss_coef = 1
_C.LOSS.cls_loss_coef = 2
_C.LOSS.bbox_loss_coef = 5
_C.LOSS.giou_loss_coef = 2
_C.LOSS.focal_alpha = 0.25

# these are for TransCenter todo
_C.LOSS.hm_weight = 0.
_C.LOSS.off_weight = 0.
_C.LOSS.wh_weight = 0.
_C.LOSS.boxes_weight = 0.
_C.LOSS.giou_weight = 0.
_C.LOSS.ct_offset_weight = 0.



_C.DATASET = CN()

_C.DATASET.cache_dir = "./datasets/cache"
_C.DATASET.img_dir = "../datasets/ILSVRC2015/Data/{}",
_C.DATASET.anno_path = "../datasets/ILSVRC2015/Annotations/{}",
_C.DATASET.img_index = "./datasets/split_file/{}.txt",
_C.DATASET.dataset_file = ["DET_train_30classes", "VID_train_15frames"]  # vid coco
_C.DATASET.val_dataset = "VID_val_videos"
_C.DATASET.coco_path = "../datasets/coco"
_C.DATASET.coco_panoptic_path = None  # todo
_C.DATASET.remove_difficult = True
_C.DATASET.num_classes = 1
_C.DATASET.input_h = 640
_C.DATASET.input_w = 1088
_C.DATASET.down_ratio = 4
_C.DATASET.output_h = 160
_C.DATASET.output_w = 272
_C.DATASET.K = 300
_C.DATASET.not_rand_crop = False
_C.DATASET.not_max_crop = False
_C.DATASET.shift = 0.05
_C.DATASET.scale = 0.05
_C.DATASET.rotate = 0
_C.DATASET.flip = 0.5
_C.DATASET.no_color_aug = False
_C.DATASET.aug_rot = 0
_C.DATASET.image_blur_aug = False
_C.DATASET.heads = ['hm', 'reg', 'wh', 'center_offset']
_C.DATASET.dense_reg = 1
_C.DATASET.debug = True
_C.DATASET.max_offset = 12
_C.DATASET.min_offset = -12
_C.DATASET.ref_num_local = 2
# _C.DATASET.scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
