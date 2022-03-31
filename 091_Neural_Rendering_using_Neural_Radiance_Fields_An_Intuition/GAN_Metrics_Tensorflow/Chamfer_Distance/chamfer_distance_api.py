import os
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.FATAL)
nn_distance_module=tf.load_op_library('./chamfer_distance_api/chamfer-distance/tf_nndistance_so.so')

def nn_distance(xyz1,xyz2):
    '''
Computes the distance of nearest neighbors for a pair of point clouds
input: xyz1: (batch_size,#points_1,3)  the first point cloud
input: xyz2: (batch_size,#points_2,3)  the second point cloud
output: dist1: (batch_size,#point_1)   distance from first to second (squared)
output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
output: dist2: (batch_size,#point_2)   distance from second to first (squared)
output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
    '''
    return nn_distance_module.nn_distance(xyz1,xyz2)

class Chamfer_distance():
    def __init__(self, verbose = False):
        '''
        Constructor of the Chamfer_distance calculator class. 
        INPUT: verbose: print messages for debug
        '''
        t_prepare_begin = time.time()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inp1=tf.placeholder(shape=[None, None, 3], dtype=tf.float32)
            self.inp2=tf.placeholder(shape=[None, None, 3], dtype=tf.float32)
            self.dist1,self.idx1,self.dist2,self.idx2=nn_distance(self.inp1,self.inp2)
            self.dist1_avg = tf.reduce_mean(self.dist1,axis=1)
            self.dist2_avg = tf.reduce_mean(self.dist2,axis=1)
            self.distance= self.dist1_avg + self.dist2_avg
        t_prepare_end = time.time()
        if verbose:
            print('time to initialize: {}'.format(t_prepare_end-t_prepare_begin))
    def get_chamfer_distance(self, xyz1, xyz2, verbose = False):
        '''
        input:  xyz1: (batch_size,#points_1,3)  the first point cloud
                xyz2: (batch_size,#points_2,3)  the second point cloud
        '''
        if xyz1.ndim == 2 and xyz2.ndim == 2:
            # add batch_size
            xyz1 = np.expand_dims(xyz1,0)
            xyz2 = np.expand_dims(xyz2,0)
        t0=time.time()
        with tf.Session(graph=self.graph) as sess:
            cd=sess.run(self.distance, feed_dict={self.inp1: xyz1, self.inp2: xyz2})
        newt=time.time()
        if verbose:
            print 'time to calculate cd: {}'.format(newt-t0)
        return cd
    
def demo():

    import random
    cd_api = Chamfer_distance(verbose = True)
    # Calculate chamfer distance between point (1,1,1) and (1,1,-1), which should output [8.].
    xyz1 = np.array([[[1,1,1]]])
    xyz2 = np.array([[[1,1,-1]]])

    # xyz1 = np.array([[[1,1,1],[-1,-1,-1]],[[1,1,1],[-1,-1,-1]]])
    # xyz2 = np.array([[[2,1,1],[-2,-1,-1]],[[2,1,1],[-2,-1,-1]]])

    # xyz1 = np.random.rand(20000,3)
    # xyz2 = np.random.rand(10000,3)
    print(cd_api.get_chamfer_distance(xyz1,xyz2, True))
if __name__=='__main__':
    demo()

