import h5py
import numpy as np
from utils.pointclouds_utils import frames_uv

import vispy
from vispy.scene import visuals

from vispy import scene
from functools import partial
from utils.pointclouds_utils import fps
import tensorflow as tf

# X = h5py.File('C:/Users/adrien/Documents/Datasets/ModelNet40_hdf5/modelnet40_hdf5_2048_original/data_hdf5/test_data_0.h5', 'r')
# print(list(X.keys()))

def setup_pcl_viewer_(X, color=(1, 1., 1, .5), run=False):
    # setup a point cloud viewer using vispy and return a drawing function
    # make a canvas and add simple view
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor=(1, 1, 1, 1))
    # canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    # create scatter object and fill in the data
    # init_pc = np.random.normal(size=(100, 3), scale=0.2)
    init_pc = X
    scatter = visuals.Markers()
    draw_fn = partial(scatter.set_data, edge_color=None, face_color=color, size=10)
    draw_fn(init_pc)
    view.add(scatter)
    # set camera
    view.camera = 'turntable'  # ['turntable','arcball']
    # add a colored 3D axis for orientation
    # axis = visuals.XYZAxis(parent=view.scene)
    return draw_fn

def tf_fibonnacci_sphere_sampling(num_pts):
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    S2 = np.stack([x, y, z], axis=-1)
    return tf.convert_to_tensor(S2, dtype=tf.float32)

"""
X = h5py.File('C:/Users/adrien/Documents/Keras/HGEN/preds/auto_encoder_epoch_50_time_30_01_2020_23_52_29.h5', 'r')
print(list(X.keys()))
i = 10
input_ = X['input'][i]
print('input shape ', input_.shape)
downsampled = X['downsampled'][i]
print('downsampled shape ', downsampled.shape)
# upsampled, _ = fps(X['upsampled'][i], input_.shape[0])
upsampled = X['upsampled'][i]
print('upsampled shape ', upsampled.shape)

setup_pcl_viewer_(input_)
# vispy.app.run()

setup_pcl_viewer_(downsampled)
# vispy.app.run()

setup_pcl_viewer_(upsampled)

"""
S = tf_fibonnacci_sphere_sampling(256)
S = np.array(S)
setup_pcl_viewer_(S, color=(0., 0., 1., .9))
vispy.app.run()





