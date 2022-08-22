_base_ = '../default.py'

expname = 'drums'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/drums',
    dataset_type='blender',
    white_bkgd=True,
)

