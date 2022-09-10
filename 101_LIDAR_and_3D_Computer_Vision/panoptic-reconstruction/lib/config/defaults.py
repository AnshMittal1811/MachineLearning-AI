import os

from yacs.config import CfgNode as Node

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
_C = Node()
_C.MODEL = Node()
_C.MODEL.PRETRAIN = ""
_C.MODEL.PRETRAIN2D = ""
_C.MODEL.DEVICE = "cuda"
_C.MODEL.FIX2D = False
_C.MODEL.FIXNORM = False

_C.MODEL.PANOPTIC_RESOLUTION = 1.0

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = Node()
_C.MODEL.BACKBONE.CONV_BODY = "R-18"
_C.MODEL.BACKBONE.PRETRAIN = True

# ---------------------------------------------------------------------------- #
# Depth2D options
# ---------------------------------------------------------------------------- #
_C.MODEL.DEPTH2D = Node()
_C.MODEL.DEPTH2D.USE = True
_C.MODEL.DEPTH2D.FIX = False
_C.MODEL.DEPTH2D.SAMPLING = "nearest"  # nearest, bilinear
_C.MODEL.DEPTH2D.LOSS_WEIGHT = 1.0


# ---------------------------------------------------------------------------- #
# Instance2D options
# ---------------------------------------------------------------------------- #
_C.MODEL.INSTANCE2D = Node()
_C.MODEL.INSTANCE2D.USE = True
_C.MODEL.INSTANCE2D.FIX = False
_C.MODEL.INSTANCE2D.MAX = 15
_C.MODEL.INSTANCE2D.MIN_PIXELS = 200
_C.MODEL.INSTANCE2D.FPN = False
_C.MODEL.INSTANCE2D.GT_PROPOSAL = False
_C.MODEL.INSTANCE2D.LOSS_WEIGHT = 1.0
# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.INSTANCE2D.RPN = Node()
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.INSTANCE2D.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
_C.MODEL.INSTANCE2D.RPN.ANCHOR_STRIDE = (16,)
# _C.MODEL.INSTANCE2D.RPN.ANCHOR_STRIDE = (4, 8, 16, 32, 64)

# RPN anchor aspect ratios
_C.MODEL.INSTANCE2D.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.INSTANCE2D.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.INSTANCE2D.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.INSTANCE2D.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.INSTANCE2D.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.INSTANCE2D.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
_C.MODEL.INSTANCE2D.RPN.PRE_NMS_TOP_N_TRAIN = 500
_C.MODEL.INSTANCE2D.RPN.PRE_NMS_TOP_N_TEST = 250
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.INSTANCE2D.RPN.POST_NMS_TOP_N_TRAIN = 500
_C.MODEL.INSTANCE2D.RPN.POST_NMS_TOP_N_TEST = 250
# NMS threshold used on RPN proposals
_C.MODEL.INSTANCE2D.RPN.NMS_THRESH = 0.3
_C.MODEL.INSTANCE2D.RPN.SCORE_THRESH = 0.5
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.INSTANCE2D.RPN.MIN_SIZE = 0

# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.INSTANCE2D.ROI_HEADS = Node()
_C.MODEL.INSTANCE2D.ROI_HEADS.USE = True
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.INSTANCE2D.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.INSTANCE2D.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.INSTANCE2D.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.INSTANCE2D.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.INSTANCE2D.ROI_HEADS.POSITIVE_FRACTION = 0.25
# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_C.MODEL.INSTANCE2D.ROI_HEADS.SCORE_THRESH = 0.3
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.INSTANCE2D.ROI_HEADS.NMS = 0.3
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.INSTANCE2D.ROI_HEADS.DETECTIONS_PER_IMG = 100

_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD = Node()
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.PREDICTOR = "FastRNodeNPredictor"
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.POOLER_SCALES = (0.0625,)
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.DILATION = 1
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4
# _C.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.NUM_CLASSES = 81
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_BOX_HEAD.NUM_CLASSES = 13

_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD = Node()
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.USE = True
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.PREDICTOR = "MaskRNodeNC4Predictor"
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 2
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.POOLER_SCALES = (0.0625,)
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = False
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
_C.MODEL.INSTANCE2D.ROI_HEADS.ROI_MASK_HEAD.DILATION = 1

# -----------------------------------------------------------------------------
# Projection options
# -----------------------------------------------------------------------------
_C.MODEL.PROJECTION = Node()
_C.MODEL.PROJECTION.USE = True
_C.MODEL.PROJECTION.SIGN_CHANNEL = True  # if true use separate channel to encode sign for sdf projection
_C.MODEL.PROJECTION.FILTER_ROOM_PIXELS = False
_C.MODEL.PROJECTION.VOXEL_SIZE = 0.03
_C.MODEL.PROJECTION.DEPTH_MIN = 0.4
_C.MODEL.PROJECTION.DEPTH_MAX = 6.0

_C.MODEL.PROJECTION.FOCAL_LENGTH = 277.1281435

_C.MODEL.PROJECTION.INTRINSIC = [[277.1281435,   0.       , 159.5,  0.],
                                 [  0.       , 277.1281435, 119.5,  0.],
                                 [  0.       ,   0.       ,   1. ,  0.],
                                 [  0.       ,   0.       ,   0. ,  1.]]  # Fixed 3D-Front intrinsic
# -----------------------------------------------------------------------------
# Frustum3d options
# -----------------------------------------------------------------------------
_C.MODEL.FRUSTUM3D = Node()
_C.MODEL.FRUSTUM3D.USE = True
_C.MODEL.FRUSTUM3D.FIX = False
_C.MODEL.FRUSTUM3D.REPRESENTATION = "df"  # df, fdf, sdf
_C.MODEL.FRUSTUM3D.GRID_DIMENSIONS = [256, 256, 256]  # df, fdf, sdf

_C.MODEL.FRUSTUM3D.ISO_VALUE = 1.0

_C.MODEL.FRUSTUM3D.NUM_CLASSES = 13

_C.MODEL.FRUSTUM3D.DENSE_SPARSE_THRESHOLD = 0.5
_C.MODEL.FRUSTUM3D.SPARSE_THRESHOLD_128 = 0.5
_C.MODEL.FRUSTUM3D.SPARSE_THRESHOLD_256 = 0.5


_C.MODEL.FRUSTUM3D.TRUNCATION = 3.0
_C.MODEL.FRUSTUM3D.UNET_OUTPUT_CHANNELS = 16
_C.MODEL.FRUSTUM3D.UNET_FEATURES = 16

# Loss weighting flags
_C.MODEL.FRUSTUM3D.COMPLETION_VOXEL_WEIGHTING = False
_C.MODEL.FRUSTUM3D.COMPLETION_128_VOXEL_WEIGHTING = False
_C.MODEL.FRUSTUM3D.COMPLETION_256_VOXEL_WEIGHTING = False
_C.MODEL.FRUSTUM3D.SURFACE_VOXEL_WEIGHTING = True
_C.MODEL.FRUSTUM3D.SEMANTIC_VOXEL_WEIGHTING = True
_C.MODEL.FRUSTUM3D.INSTANCE_VOXEL_WEIGHTING = True

_C.MODEL.FRUSTUM3D.COMPLETION_WEIGHT = 25.0
_C.MODEL.FRUSTUM3D.COMPLETION_128_WEIGHT = 10.0
_C.MODEL.FRUSTUM3D.COMPLETION_256_WEIGHT = 5.0

_C.MODEL.FRUSTUM3D.L1_WEIGHT = 5.0
_C.MODEL.FRUSTUM3D.INSTANCE_WEIGHT = 25.0
_C.MODEL.FRUSTUM3D.SEMANTIC_WEIGHT = 10.0

_C.MODEL.FRUSTUM3D.GEOMETRY_HEAD = Node()
_C.MODEL.FRUSTUM3D.GEOMETRY_HEAD.RESNET_BLOCKS = 1

_C.MODEL.FRUSTUM3D.SEMANTIC_HEAD = Node()
_C.MODEL.FRUSTUM3D.SEMANTIC_HEAD.RESNET_BLOCKS = 1


_C.MODEL.FRUSTUM3D.IS_LEVEL_64 = True
_C.MODEL.FRUSTUM3D.LEVEL_ITERATIONS_64 = 10000
_C.MODEL.FRUSTUM3D.IS_LEVEL_128 = True
_C.MODEL.FRUSTUM3D.LEVEL_ITERATIONS_128 = 5000
_C.MODEL.FRUSTUM3D.IS_LEVEL_256 = True
_C.MODEL.FRUSTUM3D.LEVEL_ITERATIONS_256 = 5000

_C.MODEL.FRUSTUM3D.IS_SDF = False

_C.MODEL.FRUSTUM3D.REWEIGHT_CLASSES = True

# 3D-Front class weights
_C.MODEL.FRUSTUM3D.CLASS_WEIGHTS = [0.001, 5.012206425, 3.67783818, 4.255409701, 4.810809432, 4.542602089,
                                    6.743836113, 4.789477957, 10.28815007, 5.401373831, 1.7803661289, 1.257145849,
                                    8.397596223]

_C.MODEL.FRUSTUM3D.IS_LEVEL_64 = True
_C.MODEL.FRUSTUM3D.IS_LEVEL_128 = True
_C.MODEL.FRUSTUM3D.IS_LEVEL_256 = True
_C.MODEL.FRUSTUM3D.FULL = True

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = Node()
_C.DATASETS.NAME = "front3d"
_C.DATASETS.NUM_TRAIN = 0
_C.DATASETS.NUM_TEST = 0
_C.DATASETS.NUM_LOAD_TEST = 0

_C.DATASETS.NUM_VALIDATE = 10
_C.DATASETS.OVERFITTING = False
_C.DATASETS.MAPPING = "datagen/front3d/nyu40labels_suncg.csv"

_C.DATASETS.FRUSTUM_DIMENSIONS = [256, 256, 256]

_C.DATASETS.FIELDS = ["color", "depth", "instances2d", "geometry"]

# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ""
_C.DATASETS.VAL = ""
_C.DATASETS.TRAINVAL = ""
_C.DATASETS.TEST = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = Node()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.IMS_PER_BATCH = 1
_C.DATALOADER.SHUFFLE = True
_C.DATALOADER.AUGMENTATION = False
_C.DATALOADER.MAX_ITER = 400000

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = Node()
_C.SOLVER.ADAM = True

_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.BETA_1 = 0.9
_C.SOLVER.BETA_2 = 0.999
_C.SOLVER.WEIGHT_DECAY = 0.00005

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.CHECKPOINT_PERIOD = 2500
_C.SOLVER.EVALUATION_PERIOD = 10000
_C.SOLVER.LOAD_SCHEDULER = True
_C.SOLVER.LOAD_OPTIMIZER = True
_C.SOLVER.QUICK_START = False

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = Node()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
_C.OUTPUT_DIR = "."
