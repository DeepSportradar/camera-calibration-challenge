from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = "ResNet18"
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.LOSS = "cross_entropy"
_C.MODEL.SEGMENTATION_LOSS = False
_C.MODEL.LOSS_FUNCTION = "loss_fn_seg"
_C.MODEL.LOSS_WEIGHT_BACKGROUND = 1.0

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = 32
# Size of the image during test
_C.INPUT.SIZE_TEST = 32
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [
    0.1307,
]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [
    0.3081,
]
# Size of the generated image from View dataset (width, height)
_C.INPUT.GENERATED_VIEW_SIZE = [500, 500]
# Range of the generated image of defined pixels per meter
_C.INPUT.GENERATED_DEF_PM = [20, 60]
# This controls the generated View size and the DEF_PM together
# Then final resolution would be MULTIPLICATIVE_FACTOR * GENERATED_VIEW_SIZE
# Then final def_pm would be MULTIPLICATIVE_FACTOR * GENERATED_DEF_PM
_C.INPUT.MULTIPLICATIVE_FACTOR = 1
# Wether to apply transforms
_C.INPUT.TRANSFORMS = True

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ""
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ""
# Number of elements to consider for training
_C.DATASETS.NUM_ELEMENTS = 0
# Enables evaluation mode
_C.DATASETS.EVALUATION = False
# Evaluation on val or test split
_C.DATASETS.EVAL_ON = "val"
# Evaluation on val or test split
_C.DATASETS.RUN_METRICS = False


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_EPOCHS = 50

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.AMSGRAD = False

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 1

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"
