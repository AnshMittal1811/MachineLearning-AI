from yacs.config import CfgNode as CN

# For clarity, this default config corresponds to Deformable DETR
_C = CN()
# ---------------------------------------------------------------------------- #
# Deformable DETR GENERAL CONFIG
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
# Path to a checkpoint file to be loaded to the model
_C.MODEL.WEIGHTS = ''
# Shifts class embed neurons to start with label=0 at neuron=0. Used to convert weights from deformable_detr official repo to this one
_C.MODEL.SHIFT_CLASS_NEURON = False
# Backbone used, available options are: 'resnet50' 'resnet101'
_C.MODEL.BACKBONE = 'resnet50'
# If true, the stride is replaced with dilation in the last convolutional block (DC5
_C.MODEL.BACKBONE_DILATION = False
# Number of query slots
_C.MODEL.NUM_QUERIES = 300
# Size of the embeddings (dimension of the transformer)
_C.MODEL.HIDDEN_DIM = 256
# Intermediate size of the feedforward layers in the transformer block
_C.MODEL.DIM_FEEDFORWARD = 1024
# Dropout applied in the transformer
_C.MODEL.DROPOUT = 0.1
# Number of feature level resolutions. The resolutions used for each number is the following:
# [/64, /32,  /16, /8] for 4 levels,  [/32,  /16, /8] for 3 levels,  [/16,
# /8] for 2 levels and [/32] with a single level. The latter allows replicating DETR resolution with deformable attention.
_C.MODEL.NUM_FEATURE_LEVELS = 4
# Enable bounding box refine at each decoder layer
_C.MODEL.WITH_BBX_REFINE = True
# Allows gradient propagation of the bounding box prediction of each decoder layer, only used when WITH_BBX_REFINE=True
_C.MODEL.BBX_GRADIENT_PROP = False
# Allows using reference point refine, which works similar as bbx refine but predicts only x,y
# offset position at each decoder layer, instead of predicting a bounding box offsets
# If active, then WITH_BBX_REFINE must be set to False
_C.MODEL.WITH_REF_POINT_REFINE = False
# Activates mask head on top of deformable detr
_C.MODEL.MASK_ON = False

# ---------------------------------------------------------------------------- #
# Deformable DETR TRANSFORMER CONFIG
# ---------------------------------------------------------------------------- #
_C.MODEL.TRANSFORMER = CN()
# Number of encoding layers in the transformer
_C.MODEL.TRANSFORMER.ENCODER_LAYERS = 6
# Number of decoding layers in the transformer
_C.MODEL.TRANSFORMER.DECODER_LAYERS = 6
# Number of attention heads inside the transformer's attentions
_C.MODEL.TRANSFORMER.N_HEADS = 8
# Number of attention points per head and per resolution used for encoder/decoder deformable
# attention. This corresponds to current frame points for DeVIS
_C.MODEL.TRANSFORMER.ENC_N_POINTS = 4
_C.MODEL.TRANSFORMER.DEC_N_POINTS = 4


# ---------------------------------------------------------------------------- #
# Deformable DETR Mask Head config
# ---------------------------------------------------------------------------- #
_C.MODEL.MASK_HEAD = CN()
# Whether to use or not modulated deformable convolutions  https://arxiv.org/abs/1811.11168 on the mask head. If not regular 2D conv are used
_C.MODEL.MASK_HEAD.USE_MDC = True
# Upsampling resolution on the mask head for which multi-head attention is computed
_C.MODEL.MASK_HEAD.UPSAMPLING_RESOLUTIONS = ['/32', '/16', '/8']
# Used features in each of the upsampling step of the mask head. NUM_FEATURE_LEVELS needs also to
# be set up accordingly . Available feats for each resolution:
# /64 : [encoded, compressed_backbone]
# /32: [encoded, compressed_backbone, backbone]
# /16: [encoded, compressed_backbone, backbone]
# /8: [encoded, compressed_backbone, backbone]
# /4: [backbone]
# 'encoded' features are from the transformer encoder output
# 'compressed_backbone' features are from the output of the input_proj layer, applied to reduce backbone  features' out channels
# 'backbone' are features from the backbone
_C.MODEL.MASK_HEAD.USED_FEATURES = [['/32', 'encoded'], ['/16', 'encoded'], ['/8', 'encoded'], ['/4', 'backbone']]
# 3D convolutional head to replicate VisTr
_C.MODEL.MASK_HEAD.DEVIS = CN()
_C.MODEL.MASK_HEAD.DEVIS.CONV_HEAD_3D = False


# ---------------------------------------------------------------------------- #
# DeVIS GENERAL CONFIG
# ---------------------------------------------------------------------------- #
_C.MODEL.DEVIS = CN()
# Number of frames used, this value is fixed for both train and test
_C.MODEL.DEVIS.NUM_FRAMES = 6
# Type of temporal positional embedding to use on top of x,y sine positional encoding.  Available
# options are: 'sine' 'learned'
_C.MODEL.DEVIS.TEMPORAL_EMBEDDING = 'learned'

# ---------------------------------------------------------------------------- #
# DeVIS TEMPORAL DEFORMABLE ATTENTION CONFIG
# ---------------------------------------------------------------------------- #
# TODO: Allow to construct decoder attention mechanism module by module
_C.MODEL.DEVIS.DEFORMABLE_ATTENTION = CN()
# Deactivates temporal connections, so deformable attention run per-frame independently.
# For ablation purposes only.
_C.MODEL.DEVIS.DEFORMABLE_ATTENTION.DISABLE_TEMPORAL_CONNECTIONS = False
# Allows activating the temporal deformable attention connections in all input frames, same as decoder
_C.MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_CONNECT_ALL_FRAMES = True
# Allows to select the window [T-W/2, T+W/2] that each frame T can sample from in the encoder. If
# ENC_CONNECT_ALL_FRAMES=True this value is ignored
_C.MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_TEMPORAL_WINDOW = 4
# Activates instance aware attention on the decoder, enabling to sample from same instance
# position on other frames and use same instance bbx in order to modulate sampling offsets
_C.MODEL.DEVIS.DEFORMABLE_ATTENTION.INSTANCE_AWARE_ATTENTION = True
# Number of points from the encoder/decoder's temporal deformable attention for each attention
# head, each resolution and each frame.
_C.MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_N_POINTS_TEMPORAL_FRAME = 4
_C.MODEL.DEVIS.DEFORMABLE_ATTENTION.DEC_N_POINTS_TEMPORAL_FRAME = 4


# ---------------------------------------------------------------------------- #
# LOSS WEIGHTS
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSS = CN()
# Activates auxiliary loss
_C.MODEL.LOSS.AUX_LOSS = True
# Activates auxiliary loss weighting strategy, which follows : Layer 1: 1/30, Layer 2: 2/30,
# Layer 3: 3/30,  Layer 4: 4/30   Layer 5: 5/30  Layer 6: 15/30. These coefficients are NOT
# applied to MASK_AUX_LOSS if activated
_C.MODEL.LOSS.AUX_LOSS_WEIGHTING = False
# Activates focal loss as originally used by Deformable DeTR. If False Softmax loss is used,
# same as in DETR / VisTR
_C.MODEL.LOSS.FOCAL_LOSS = True
# Activate mask auxiliary loss at the output of the desired layer. If list is empty mask aux loss
# is not computed
_C.MODEL.LOSS.MASK_AUX_LOSS = [2, ]
# Sigmoid focal loss mask coefficient
_C.MODEL.LOSS.SEGM_MASK_COEF = 1.0
# Dice loss mask coefficient
_C.MODEL.LOSS.SEGM_DICE_COEF = 1.0
# L1 bbx  loss  coefficient
_C.MODEL.LOSS.BBX_L1_COEF = 5.0
# GIOU bbx loss  coefficient
_C.MODEL.LOSS.BBX_GIOU_COEF = 2.0
# Classification loss  coefficient
_C.MODEL.LOSS.CLASS_COEF = 2.0
# Focal alpha coefficient for focal loss on classification
_C.MODEL.LOSS.FOCAL_ALPHA = 0.25
# Relative classification weight of the no-object class when FOCAL_LOSS=False
_C.MODEL.LOSS.EOS = 0.1


# ---------------------------------------------------------------------------- #
# MATCHER
# ---------------------------------------------------------------------------- #
_C.MODEL.MATCHER = CN()
_C.MODEL.MATCHER.CLASS_COST = 2.0
_C.MODEL.MATCHER.BBX_L1_COST = 5.0
_C.MODEL.MATCHER.BBX_GIOU_COST = 2.0
# TODO: Try removing this, as only applies to DeVIS and is inherit from VisTR and not yet tested
_C.MODEL.MATCHER.USE_SUM_L1_DISTANCE = False


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Allows switching between the VIS or Instance segmentation model. Allowed values: 'vis' or 'coco'
_C.DATASETS.TYPE = 'coco'
# Root path of all training data.
_C.DATASETS.DATA_PATH = 'data'
_C.DATASETS.TRAIN_DATASET = 'train'
_C.DATASETS.VAL_DATASET = 'val'
_C.DATASETS.DEVIS = CN()
_C.DATASETS.DEVIS.COCO_JOINT_TRAINING = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Allows scaling by a factor all the different scales involved on the training transformation pipeline.
_C.INPUT.SCALE_FACTOR_TRAIN = 1.0
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333

_C.INPUT.DEVIS = CN()
# Activates multi-scale input training for DeVIS (always true for Deformable DeTr). Otherwise,
# we use the transformation pipeline as VisTr, which crops always the image and resizes it to a
# fixed final shape.
_C.INPUT.DEVIS.MULTI_SCALE_TRAIN = True
# Allows sampling each frame of each video in the dataset as starting clip_length sub-clip. If
# False we only until the VIDEO_SIZE- DEVIS.NUM_FRAMES th frame.
_C.INPUT.DEVIS.SAMPLE_EACH_FRAME = False
# Allows re-computing the bounding box coordinates using the resulting mask after all the
# training data transformations
_C.INPUT.DEVIS.CREATE_BBX_FROM_MASK = True


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 0.0002

# Parameters to freeze
_C.SOLVER.FROZEN_PARAMS = []

# Backbone is trained with a decreased lr factor
_C.SOLVER.BACKBONE_NAMES = ['backbone.0']
_C.SOLVER.LR_BACKBONE = 0.00002

# Lr from deformable attention temporal linear layer, which applies to CURRENT_FRAME for VIS
# training
_C.SOLVER.LR_LINEAR_PROJ_NAMES = ['self_attn.sampling_offsets', 'cross_attn.sampling_offsets', 'reference_points']
_C.SOLVER.LR_LINEAR_PROJ_MULT = 0.1

# Lr from mask head, needs to be higher when training inst segm model
_C.SOLVER.LR_MASK_HEAD_NAMES = ['bbox_attention', 'mask_head']
_C.SOLVER.LR_MASK_HEAD_MULT = 1

# Lr from temporal deformable attention linear layer introduced with the VIS training
_C.SOLVER.DEVIS = CN()
_C.SOLVER.DEVIS.LR_TEMPORAL_LINEAR_PROJ_NAMES = ['temporal_sampling_offsets', ]
_C.SOLVER.DEVIS.LR_TEMPORAL_LINEAR_PROJ_MULT = 0.1
_C.SOLVER.DEVIS.FINETUNE_QUERY_EMBEDDINGS = False
_C.SOLVER.DEVIS.FINETUNE_TEMPORAL_MODULES = True
_C.SOLVER.DEVIS.FINETUNE_CLASS_LOGITS = False


# Start epoch allows to change the start epochs when a module is resumed, so it doesn't use the last one.
_C.START_EPOCH = 1
# Total number of epochs
_C.SOLVER.EPOCHS = 50
# Epoch number to decrease learning rate by GAMMA
_C.SOLVER.STEPS = [40, ]

# Batch size for instance segmentation training, VIS model only works with BATCH_SIZE=1
_C.SOLVER.BATCH_SIZE = 2

_C.SOLVER.GAMMA = 0.1


_C.SOLVER.WEIGHT_DECAY = 0.0001

# Allows resuming optimizer from the checkpoint model
_C.SOLVER.RESUME_OPTIMIZER = False


# Epoch interval for model saving, if 0 only save last model
_C.SOLVER.CHECKPOINT_INTERVAL = 1
# Gradient clipping max norm
_C.SOLVER.GRAD_CLIP_MAX_NORM = 0.1

# ---------------------------------------------------------------------------- #
# Test option used for both eval during training and test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Epoch interval for model eval during training. If 0 doesnt run eval.
_C.TEST.EVAL_PERIOD = 1
# Epoch to start evaluation the model if epoch % EVAL_PERIOD
_C.TEST.START_EVAL_EPOCH = 1

# Folder name where results are saved, create in OUTPUT_DIR
_C.TEST.SAVE_PATH = 'eval_results'
# Number of output predictions. If TEST.CLIP_TRACKING.USE_TOP_K=False this can only be <= NUM_QUERIES per frame
_C.TEST.NUM_OUT = 100
# Set to False deactivate TopK and output max confident category for each instance only.
_C.TEST.USE_TOP_K = True

# Epoch interval for model eval during training. If 0 doesnt run eval.
_C.TEST.CLIP_TRACKING = CN()
# Clip tracking stride used, this is, number of non-overlapping frames that the model process at each step.
_C.TEST.CLIP_TRACKING.STRIDE = 4
# Allows running hungarian optimal assignment between instances with the same category only, becoming thus TEST.CLIP_TRACKING.CLASS_COST irrelevant
_C.TEST.CLIP_TRACKING.PER_CLASS_MATCHING = False
# Change Soft-IoU for Mask-IoU with the final masks of each instance
_C.TEST.CLIP_TRACKING.USE_BINARY_MASK_IOU = False
# Compute Mask-cost (soft-iou /mask-iou) averaging per-frame values instead of volumetric
_C.TEST.CLIP_TRACKING.USE_FRAME_AVERAGE_IOU = False
# Allows specifying how the final class for an instance is computed after all sub-clips results accumulation. Possible values: ['most_common', 'score_weighting']
_C.TEST.CLIP_TRACKING.FINAL_CLASS_POLICY = 'most_common'
# Allows specifying how the final score for an instance is computed after all sub-clips results accumulation. Possible values: ['mean', 'median']
_C.TEST.CLIP_TRACKING.FINAL_SCORE_POLICY = 'mean'
# Classification cost weight used to penalize matching between instances with different categories
_C.TEST.CLIP_TRACKING.CLASS_COST = 1
# Soft-IoU/Mask-IoU cost weight to determine similarity between different predicted masks
_C.TEST.CLIP_TRACKING.MASK_COST = 1
# Score cost applied to encourage matching between high-high and low-low confidence instances
_C.TEST.CLIP_TRACKING.SCORE_COST = 1
# Embedding center distance cost applied to reward spatially close instances, specially useful when there isn't mask prediction (0s).
_C.TEST.CLIP_TRACKING.CENTER_COST = 0
# Removes output segmentation mask of a particular frame if score below this threshold
_C.TEST.CLIP_TRACKING.MIN_FRAME_SCORE = 0.001
# Removes output instance if final score is below this threshold
_C.TEST.CLIP_TRACKING.MIN_TRACK_SCORE = 0.002
# Removes output instance if number of not None mask predictions is below this threshold
_C.TEST.CLIP_TRACKING.MIN_DETECTIONS = 1

# Allows evaluating epochs specified in EPOCHS_TO_EVAL from this folder. Checkpoint names must follow default format written during training.
_C.TEST.INPUT_FOLDER = ''
# Epochs to be evaluated from TEST.INPUT_FOLDER
_C.TEST.EPOCHS_TO_EVAL = [6, 7, 8, 9, 10]

_C.TEST.VIZ = CN()
# Path to save output results visualization. If specified results will be generated in this folder
_C.TEST.VIZ.OUT_VIZ_PATH = ''
# Allows saving results obtained in each sub-clip. OUT_VIZ_PATH  needs to be specified
_C.TEST.VIZ.SAVE_CLIP_VIZ = False
# Allows saving results plotting all instances on the same image. OUT_VIZ_PATH  needs to be specified
_C.TEST.VIZ.SAVE_MERGED_TRACKS = False
# Allows running only the desired video names from all the validation dataset videos.
# Must be separated by comma.
_C.TEST.VIZ.VIDEO_NAMES = ''

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Dataloader number of workers used.
_C.NUM_WORKERS = 4
# Directory where output files are written
_C.OUTPUT_DIR = './output'
# Interval for log output and Visdom
_C.VISDOM_AND_LOG_INTERVAL = 100
# Activates visdom visualization
_C.VISDOM_ON = True
# TODO: Check RESUME_VIS because it didnt seem to work properly, visdom was resumed at the correct epoch but previous info was removed
_C.RESUME_VIS = False
_C.VISDOM_PORT = 8090
_C.VISDOM_SERVER = 'http://localhost'
_C.SEED = 42
_C.DEVICE = 'cuda'


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
