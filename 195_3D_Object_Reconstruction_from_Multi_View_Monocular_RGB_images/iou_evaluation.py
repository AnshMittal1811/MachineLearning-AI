import tensorflow as tf
import os
import sys
sys.path.append('..')
import tools as tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import copy 
import shutil
GPU='0'
resolution = 32
batch_size = 1
vox_res = 32

def metric_IoU(batch_voxel_occup_pred, batch_voxel_occup_true):
    batch_voxel_occup_pred_ = copy.deepcopy(batch_voxel_occup_pred)
    batch_voxel_occup_pred_[batch_voxel_occup_pred_ >= 0.5] = 1
    batch_voxel_occup_pred_[batch_voxel_occup_pred_ < 0.5] = 0
	
    I = batch_voxel_occup_pred_ * batch_voxel_occup_true
    U = batch_voxel_occup_pred_ + batch_voxel_occup_true			
    U[U < 1] = 0
    U[U >= 1] = 1
    iou = np.sum(I) * 1.0 / np.sum(U) * 1.0
    return iou
	
def process_voxel(src):
    if len(src)<=0:
        
        exit()
    batch_i = 0
    dst = np.zeros((batch_size, resolution, resolution, resolution, 1))
    for batch in src:
        for i in batch:
            dst[int(batch_i), int(i[0]), int(i[1]), int(i[2]), 0] = 1 # occupied
        batch_i += 1
    return dst	

def evaluate_voxel_prediction(prediction, gt):
  #"""  The prediction and gt are 3 dim voxels. Each voxel has values 1 or 0"""
 #   prediction=prediction.astype(np.int)    
    intersection = np.sum(np.logical_and(prediction,gt))
    union = np.sum(np.logical_or(prediction,gt))
    IoU = float(intersection) / float(union)
    return IoU
  
def load_real_rgbs(test_mv=3):
    obj_rgbs_folder ='./Data_sample/amazon_real_rgbs/lamp/'
    rgbs = []
    rgbs_views = sorted(os.listdir(obj_rgbs_folder))
    for v in rgbs_views:
        if not v.endswith('png'): continue
        rgbs.append(tools.Data.load_single_X_rgb_r2n2(obj_rgbs_folder + v, train=False))
    rgbs = np.asarray(rgbs)
    x_sample = rgbs[0:test_mv, :, :, :].reshape(1, test_mv, 127, 127, 3)
    return x_sample, None

def load_shapenet_rgbs(test_mv=3):
    obj_gt_vox_path ='./Data_sample/ShapeNetVox32/03001627/1a74a83fa6d24b3cacd67ce2c72c02e/model.binvox'
    rgbs=[]
    rgbs_views = sorted(os.listdir(obj_rgbs_folder))
    for v in rgbs_views:
        if not v.endswith('png'): continue
        rgbs.append(tools.Data.load_single_X_rgb_r2n2(obj_rgbs_folder + v, train=False))
    rgbs = np.asarray(rgbs)
    x_sample = rgbs[0:test_mv, :, :, :].reshape(1, test_mv, 127, 127, 3)
    y_true = tools.Data.load_single_Y_vox(obj_gt_vox_path)
    #########################################
    Y_true_vox = []
    Y_true_vox.append(y_true)
    Y_true_vox = np.asarray(Y_true_vox)
    return x_sample, Y_true_vox
    #########################################
def load_data():

        gt_vox1=gt_vox.astype(np.float32)
        Y_vox_ = tf.reshape(gt_vox1, shape=[-1, vox_res ** 3])
        Y_pred_ = tf.reshape(Y_pred, shape=[-1, vox_res ** 3])
        rec_loss=tf.reduce_mean(-tf.reduce_mean(Y_vox_*tf.log(Y_pred_ + 1e-8),reduction_indices=[1])-tf.reduce_mean((1-Y_vox_)*tf.log(1 - Y_pred_+1e-8),reduction_indices=[1]))
        
                                #########################################################
        ## session run
        y_pred,recon_loss = sess.run([Y_pred, rec_loss], feed_dict={X: x_sample})		
		
        print("y_pred",y_pred.shape)		
        print("Cross entropy loss : ",	recon_loss)
		
        iou_=evaluate_voxel_prediction(y_pred,gt_vox1)
        print("evaluate_voxel_prediction:",iou_)
		
        iou_value= metric_IoU( y_pred,gt_vox1)
        print("metric_IoU :",iou_value)		                     
        
#        y_pred= sess.run(Y_pred, feed_dict={X: x_sample})             
    ###### to visualize
    th = 0.25
    y_pred[y_pred>=th]=1
    y_pred[y_pred<th]=0
    tools.Data.plotFromVoxels(np.reshape(y_pred,[32,32,32]), title='y_pred')
    if gt_vox is not None:
        tools.Data.plotFromVoxels(np.reshape(gt_vox,[32,32,32]), title='y_true')
    from matplotlib.pyplot import show
    show()

#########################
if __name__ == '__main__':
    ttest_demo()
