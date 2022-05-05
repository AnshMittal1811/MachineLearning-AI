"""
Rethinking Portrait Matting with Privacy Preserving
config file.

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting
Paper link: https://arxiv.org/abs/2203.16828

"""


########## Root paths and logging files paths
REPOSITORY_ROOT_PATH = ''
P3M_DATASET_ROOT_PATH = ''
RWP_TEST_SET_ROOT_PATH = ''

######### Paths of datasets
DATASET_PATHS_DICT={
'P3M10K':{
    'P3M_500_P':{
        'ROOT_PATH':P3M_DATASET_ROOT_PATH+'validation/P3M-500-P/',
        'ORIGINAL_PATH':P3M_DATASET_ROOT_PATH+'validation/P3M-500-P/blurred_image/',
        'MASK_PATH':P3M_DATASET_ROOT_PATH+'validation/P3M-500-P/mask/',
        'TRIMAP_PATH':P3M_DATASET_ROOT_PATH+'validation/P3M-500-P/trimap/',
        'PRIVACY_MASK_PATH': P3M_DATASET_ROOT_PATH+'validation/P3M-500-P/facemask/',
        'SAMPLE_NUMBER':500
        },
    'P3M_500_NP':{
        'ROOT_PATH':P3M_DATASET_ROOT_PATH+'validation/P3M-500-NP/',
        'ORIGINAL_PATH':P3M_DATASET_ROOT_PATH+'validation/P3M-500-NP/original_image/',
        'MASK_PATH':P3M_DATASET_ROOT_PATH+'validation/P3M-500-NP/mask/',
        'TRIMAP_PATH':P3M_DATASET_ROOT_PATH+'validation/P3M-500-NP/trimap/',
        'PRIVACY_MASK_PATH': None,
        'SAMPLE_NUMBER':500
        },
    },
'RWP': {
    'RWP': {
        'ROOT_PATH': RWP_TEST_SET_ROOT_PATH,
        'ORIGINAL_PATH': RWP_TEST_SET_ROOT_PATH+'image/',
        'MASK_PATH': RWP_TEST_SET_ROOT_PATH+'alpha/',
        'TRIMAP_PATH': None,
        'PRIVACY_MASK_PATH': None,
        'SAMPLE_NUMBER': 636
        }
    }
}

######### Paths of samples for test

SAMPLES_ORIGINAL_PATH = REPOSITORY_ROOT_PATH+'samples/original/'
SAMPLES_RESULT_ALPHA_PATH = REPOSITORY_ROOT_PATH+'samples/result_alpha/'
SAMPLES_RESULT_COLOR_PATH = REPOSITORY_ROOT_PATH+'samples/result_color/'

######### Paths of pretrained model
PRETRAINED_R34_MP = REPOSITORY_ROOT_PATH+'models/pretrained/'
PRETRAINED_SWIN_STEM_POOLING5 = REPOSITORY_ROOT_PATH+'models/pretrained/'
PRETRAINED_VITAE_NORC_MAXPOOLING_BIAS_BASIC_STAGE4_14 = REPOSITORY_ROOT_PATH+'models/'

######### Test config
MAX_SIZE_H = 1600
MAX_SIZE_W = 1600
MIN_SIZE_H = 512
MIN_SIZE_W = 512
SHORTER_PATH_LIMITATION = 1080
