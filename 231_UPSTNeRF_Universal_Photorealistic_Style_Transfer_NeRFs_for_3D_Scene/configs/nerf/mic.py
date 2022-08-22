_base_ = '../default.py'

expname = 'mic'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/mic',
    dataset_type='blender',
    white_bkgd=True,
)

