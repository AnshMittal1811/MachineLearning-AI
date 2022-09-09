_base_ = '../default.py'

expname = 'materials'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/materials',
    dataset_type='blender',
    white_bkgd=True,
)

