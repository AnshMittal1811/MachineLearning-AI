import tensorflow as tf
import os
import shutil
import sys
import scipy.io
sys.path.append('..')
import tools as tools
import numpy as np
import time
from keras.layers import BatchNormalization,Conv3D,MaxPooling3D,Dense,Reshape,Add,LeakyReLU,Conv3DTranspose
from keras.activations import relu,sigmoid,tanh
from keras import models
import copy 
import matplotlib.pyplot as plt 
###################
#import tflearn
###################

batch_size = 1
img_res = 127
vox_res32 = 32
total_mv = 24   
GPU0 = '0'
GPU1 = '1'

re_train=True
#re_train=True
single_view_train = True
multi_view_train = True

###########################

plot_list_iou = []
plot_list_i = []
iii=0
config={}                                 # python dictionary
config['batch_size'] = batch_size
config['total_mv'] = total_mv
config['cat_names'] = ['02691156','02828884','04530566','03636649','03001627']
#config['cat_names'] = ['02691156','02828884','02933112','02958343','03001627','03211117',
#            '03636649','03691459','04090263','04256520','04379243','04401088','04530566']
#config['cat_names'] = ['02828884']
for name in config['cat_names']:
    config['X_rgb_'+name] = '/home/wiproec4/3d reconstruction/attsets/Data_sample/shapenet dataset/ShapeNetRendering/train_1_dataset/'+name+'/'

    config['Y_vox_'+name] = '/home/gpu/Desktop/Ajith_Balakrishnan/Data_sample/ShapeNetVox32/'+name+'/'

   # config['Y_vox_'+name] = '/home/wiproec4/3d reconstruction/attsets/Data_sample/shapenet dataset/ShapeNetVox32/train_1_dataset/'+name+'/'


# output : {'batch_size': 1, 'total_mv': 24, 'cat_names': ['03001627'], 'Y_vox_03001627': '/home/wiproec4/3d reconstruction/attsets/Data_sample/#ShapeNetVox32/03001627/', 'X_rgb_03001627': '/home/wiproec4/3d reconstruction/attsets/Data_sample/ShapeNetRendering/03001627/'}

def metric_iou(prediction, gt):
#    labels = tf.greater_equal(gt[gt], 0.5)
#    prediction = tf.cast(prediction,tf.int32)
    predictions = tf.greater_equal(prediction, 0.5)
    gt_=tf.greater_equal(gt, 0.5)
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(predictions,gt_),tf.float32))
    union = tf.reduce_sum(tf.cast(tf.math.logical_or(predictions,gt_),tf.float32))
    iou = tf.cast(x = intersection,dtype=tf.float32)/ tf.cast(x = union,dtype=tf.float32)
    return iou


def graph_plot(iou_value,i_value):
    x = i_value
    y = iou_value
    plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12) 
    
    plt.ylim(0,2) 
    plt.xlim(0,500) 
 
    plt.xlabel('Iterations') 

    plt.ylabel('IOU') 
  
    plt.title('Refiner Accuracy') 
  
    plt.show() 
         

def evaluate_voxel_prediction(prediction, gt):
  #"""  The prediction and gt are 3 dim voxels. Each voxel has values 1 or 0"""
    prediction[prediction >= 0.5] = 1
    prediction[prediction < 0.5] = 0        
    intersection = np.sum(np.logical_and(prediction,gt))
    union = np.sum(np.logical_or(prediction,gt))
    IoU = float(intersection) / float(union)
    return IoU
#####################################
def refiner_network(volumes_in):
    with tf.device('/gpu:' + GPU1):
        with tf.variable_scope('ref_enc'):
	
            input_volumes_32 = tf.reshape(volumes_in, [-1, vox_res32, vox_res32, vox_res32, 1],name="ref_net_in")
	
            print("input_volumes_32_shape" , input_volumes_32.shape)   #input_volumes_32_shape (?,32,32,32,1)
	
            rn1=Conv3D(filters=32, kernel_size=(4, 4, 4), padding='same',data_format="channels_last",name='ref_c1')(input_volumes_32)
            rn2=BatchNormalization()(rn1)
            rn3=LeakyReLU(alpha=.2)(rn2)
            print("rn3.shape",rn3.shape)                              # rn3.shape (?, 32, 32, 32, 32)
            volumes_16_l =MaxPooling3D(pool_size=(2, 2, 2),name='ref_m1')(rn3)
	
            print("volumes_16_l_shape" , volumes_16_l.shape)      #volumes_16_l_shape (?,16,16,16,32)
	
            rn5=Conv3D(filters=64, kernel_size=(4, 4, 4), padding='same',data_format="channels_last",name='ref_c2')(volumes_16_l)           
            rn6=BatchNormalization()(rn5)
            rn7=LeakyReLU(alpha=.2)(rn6)
            print("rn7.shape",rn7.shape)                            #rn7.shape (?, 16, 16, 16, 64)
            volumes_8_l =MaxPooling3D(pool_size=(2, 2, 2),name='ref_m2')(rn7)
		
            print("volumes_8_l_shape" ,volumes_8_l.shape)          #volumes_8_l_shape (?,8,8,8,64)
		
            rn9=Conv3D(filters=128, kernel_size=(4, 4, 4), padding='same',data_format="channels_last",name='ref_c3')(volumes_8_l)
            rn10=BatchNormalization()(rn9)
            rn11=LeakyReLU(alpha=.2)(rn10)
            print("rn11.shape",rn11.shape)                         #rn11.shape (?, 8, 8, 8, 128)
            volumes_4_l =MaxPooling3D(pool_size=(2, 2, 2),name='ref_m3')(rn11)
		
            print("volumes_4_l_shape" , volumes_4_l.shape)    #volumes_4_l_shape (?,4,4,4,128)
		
            flatten_features=tf.reshape(volumes_4_l , [-1,8192],name="ref_fc1_in")   
        with tf.variable_scope('ref_fc'):
		
            fc1=Dense(units=2048, activation='relu',name='ref_fc1')(flatten_features)
#            fc1=tanh(fc1)
            
            fc1=relu(fc1, alpha=0.0, max_value=None, threshold=0.0)
		
            print("fc1_shape",fc1.shape)       #fc1_shape (?,4,4,4,2048)
		
            fc2=Dense(units=8192, activation='relu',name='ref_fc2')(fc1)
#            fc2=tanh(fc2)
            fc2=relu(fc2, alpha=0.0, max_value=None, threshold=0.0)
		
            print("fc2_shape",fc2.shape)      #fc2_shape (?,4,4,4,8192)
			
            fc2=tf.reshape(fc2, [-1, 4,4,4,128],name="ref_fc2_out")     
		
        with tf.variable_scope('ref_Dec'):
		
            reshaped_1=Add()([fc2,volumes_4_l]) 
		
            print("reshaped_1.shape",reshaped_1.shape) #reshaped_1.shape (?,4,4,4,128)

            rn13=Conv3DTranspose(filters=64, kernel_size=(4, 4, 4), padding='same',data_format="channels_last",name='ref_d1',strides=(2, 2, 2))(reshaped_1)
	
            rn14=BatchNormalization()(rn13)
            volumes_4_r=relu(rn14, alpha=0.0, max_value=None, threshold=0.0)
		
            print("volumes_4_r_shape",volumes_4_r.shape)  #volumes_4_r_shape (?,8,8,8,64)
		
            reshaped_2=Add() ([volumes_4_r,volumes_8_l]) 
		
            print("reshaped_2_shape",reshaped_2.shape)   #volumes_2_shape (?,8,8,8,64)

	
            rn16=Conv3DTranspose(filters=32, kernel_size=(4, 4, 4), padding='same',data_format="channels_last",name='ref_d2',strides=(2, 2, 2))(reshaped_2)
            rn17=BatchNormalization()(rn16)
            volumes_8_r =relu(rn17, alpha=0.0, max_value=None, threshold=0.0)
		 
            reshaped_3=Add()([volumes_8_r,volumes_16_l])
		
            print("reshaped_3_shape",reshaped_3.shape)    #reshaped_3_shape (?,16,16,16,32)
		
	
            rn19=Conv3DTranspose(filters=1, kernel_size=(4, 4, 4), padding='same',data_format="channels_last",name='ref_d3',strides=(2, 2, 2))(volumes_8_r)
            print("rn19_shape",rn19.shape)                      #rn19_shape (?, ?, ?, ?, 1)        
#            volumes_16_r= tf.nn.sigmoid(rn19,name='ref_sigmoid1')
#            reshape_4=volumes_16_r                             ####################

            reshape_4=Add()([rn19,input_volumes_32])
            reshape_4=(reshape_4*0.5)
            print("reshape_4_5",reshape_4.shape)       #reshape_4_5 (?,32,32,32,1)
            
            reshape_4= tf.nn.sigmoid(reshape_4,name='ref_sigmoid1')
           
		
            print("reshape_4_sig_shape",reshape_4.shape)  #reshape_4_sig_shape (?,32,32,32,1)
		
            reshape_5=tf.reshape(reshape_4, [-1, vox_res32, vox_res32, vox_res32],name="ref_out")

            return reshape_5
	
def attsets_fc(x, out_ele_num):
	with tf.variable_scope('att_fc'):
		in_ele_num = tf.shape(x)[1]
		in_ele_len = int(x.get_shape()[2])
		out_ele_len = in_ele_len    
		print("out_ele_len ", out_ele_len)
		####################
		x_1st = x
		x_1st_tp = tf.reshape(x_1st, [-1, in_ele_len],name="att_in")
		weights_1st = tools.Ops.fc(x_1st_tp, out_d=out_ele_num*out_ele_len, name="att")
		
		########## option 1
		weights_1st = weights_1st
		########## option 2
#		weights_1st = tf.nn.tanh(weights_1st)

		####################
		weights_1st = tf.reshape(weights_1st, [-1, in_ele_num, out_ele_num, out_ele_len],name="att_fc_out")
		weights_1st = tf.nn.softmax(weights_1st, 1)
		x_1st = tf.tile(x_1st[:,:,None,:], [1,1,out_ele_num,1])
		x_1st = x_1st*weights_1st
		x_1st = tf.reduce_sum(x_1st, axis=1)
		x_1st = tf.reshape(x_1st, [-1, out_ele_num*out_ele_len],name="att_out")       
		return x_1st, weights_1st

#####################################
class Network:
	def __init__(self):
		self.train_mod_dir = './train_mod/'
		self.train_sum_dir = './train_sum/'
		self.test_res_dir = './test_res/'
		self.test_sum_dir = './test_sum/'
		
		print ('re_train : ', re_train)
		if os.path.exists(self.test_res_dir):
			if re_train:
				print ('test_res_dir and files kept!')
			else:
				shutil.rmtree(self.test_res_dir)
				os.makedirs(self.test_res_dir)
				print ('test_res_dir: deleted and then created!')
		else:
			os.makedirs(self.test_res_dir)
			print ('test_res_dir: created!')
		
		if os.path.exists(self.train_mod_dir):
			if re_train:
				if os.path.exists(self.train_mod_dir + 'model.cptk.data-00000-of-00001'):
					print ('model found! will be reused!')
				else:
					print ('model not found! error!')
					#exit()
			else:
				shutil.rmtree(self.train_mod_dir)
				os.makedirs(self.train_mod_dir)
				print ('train_mod_dir: deleted and then created!')
		else:
			os.makedirs(self.train_mod_dir)
			print ('train_mod_dir: created!')
		
		if os.path.exists(self.train_sum_dir):
			if re_train:
				print ('train_sum_dir and files kept!')
			else:
				shutil.rmtree(self.train_sum_dir)
				os.makedirs(self.train_sum_dir)
				print ('train_sum_dir: deleted and then created!')
		else:
			os.makedirs(self.train_sum_dir)
			print ('train_sum_dir: created!')
		
		if os.path.exists(self.test_sum_dir):
			if re_train:
				print ('test_sum_dir and files kept!')
			else:
				shutil.rmtree(self.test_sum_dir)
				os.makedirs(self.test_sum_dir)
				print ('test_sum_dir: deleted and then created!')
		else:
			os.makedirs(self.test_sum_dir)
			print ('test_sum_dir: created!')

	def base_r2n2(self, X_rgb):
		with tf.variable_scope('Encoder'):
			im_num = tf.shape(X_rgb)[1]

			[_, _, d1, d2, cc] = X_rgb.get_shape()
			X_rgb = tf.reshape(X_rgb, [-1, int(d1), int(d2), int(cc)],name="en_in")
			print("Network Structure")
			print("base_r2n2",X_rgb.shape) #base_r2n2 (?, 127, 127, 3)
			
#			plot_buf_1= tf.reshape(X_rgb, [-1, 127, 127, 3]) ###################
#			tf.summary.image("Input", plot_buf_1)###############################
	 
			en_c = [96, 128, 256, 256, 256, 256]
			l1 = tools.Ops.xxlu(tools.Ops.conv2d(X_rgb, k=7, out_c=en_c[0], str=1, name='en_c1'), label='lrelu')
			print("l1_r2n",l1.shape) #l1_r2n (?, 127, 127, 96)


			
			l2 = tools.Ops.xxlu(tools.Ops.conv2d(l1, k=3, out_c=en_c[0], str=1, name='en_c2'), label='lrelu')
			l2 = tools.Ops.maxpool2d(l2, k=2, s=2, name='en_mp1')
			print("l2_r2n",l2.shape) #l2_r2n (?, 64, 64, 96)
			
#			plot_buf_1= tf.reshape(l2, [-1, 32, 32, 3]) #########################
#			tf.summary.image("L2_MP_en", plot_buf_1)#############################
			
			l3 = tools.Ops.xxlu(tools.Ops.conv2d(l2, k=3, out_c=en_c[1], str=1, name='en_c3'), label='lrelu')
			print("l3_r2n",l3.shape) #l3_r2n (?, 64, 64, 128)
			l4 = tools.Ops.xxlu(tools.Ops.conv2d(l3, k=3, out_c=en_c[1], str=1, name='en_c4'), label='lrelu')
			print("l4_r2n",l4.shape) #l4_r2n (?, 64, 64, 128)
			l22 = tools.Ops.conv2d(l2, k=1, out_c=en_c[1], str=1, name='en_c22')
			print("l22_r2n",l22.shape) #l22_r2n (?, 64, 64, 128)
			l4 = l4 + l22
			l4 = tools.Ops.maxpool2d(l4, k=2, s=2, name='en_mp2')
			print("l4+l22_r2n",l4.shape) #l4+l22_r2n (?, 32, 32, 128)
			
#			plot_buf_1= tf.reshape(l4, [-1, 32, 32, 3]) ##########################
#			tf.summary.image("l4+l22_en", plot_buf_1)#############################

			l5 = tools.Ops.xxlu(tools.Ops.conv2d(l4, k=3, out_c=en_c[2], str=1, name='en_c5'), label='lrelu')
			print("l5_r2n",l5.shape) #l5_r2n (?, 32, 32, 256)
			l6 = tools.Ops.xxlu(tools.Ops.conv2d(l5, k=3, out_c=en_c[2], str=1, name='en_c6'), label='lrelu')
			print("l6_r2n",l6.shape) #l6_r2n (?, 32, 32, 256)
			l44 = tools.Ops.conv2d(l4, k=1, out_c=en_c[2], str=1, name='en_c44')
			print("l44_r2n",l44.shape) #l44_r2n (?, 32, 32, 256)
			l6 = l6 + l44
			l6 = tools.Ops.maxpool2d(l6, k=2, s=2, name='en_mp3')
			print("l6+l44_r2n",l6.shape) #l6+l44_r2n (?, 16, 16, 256)
			
#			plot_buf_1= tf.reshape(l6, [-1, 16, 16, 3]) ##########################
#			tf.summary.image("l6+l44_en", plot_buf_1)#############################

			l7 = tools.Ops.xxlu(tools.Ops.conv2d(l6, k=3, out_c=en_c[3], str=1, name='en_c7'), label='lrelu')
			print("l7_r2n",l7.shape) #l7_r2n (?, 16, 16, 256)
			l8 = tools.Ops.xxlu(tools.Ops.conv2d(l7, k=3, out_c=en_c[3], str=1, name='en_c8'), label='lrelu')
			print("l8_r2n",l8.shape)#l8_r2n (?, 16, 16, 256)
#			l66=tools.Ops.conv2d(l6, k=1, out_c=en_c[3], str=1, name='en_c66')
#			print("l66_r2n",l66.shape)
#			l8=l8+l66
			l8 = tools.Ops.maxpool2d(l8, k=2, s=2, name='en_mp4')
			print("l8_r2n",l8.shape) #l8_r2n (?, 8, 8, 256)
			
#			plot_buf_1= tf.reshape(l8, [-1, 8, 8, 3]) ########################
#			tf.summary.image("l8_en", plot_buf_1)#############################

			l9 = tools.Ops.xxlu(tools.Ops.conv2d(l8, k=3, out_c=en_c[4], str=1, name='en_c9'), label='lrelu')
			print("l9_r2n",l9.shape) #l9_r2n (?, 8, 8, 256)
			l10 = tools.Ops.xxlu(tools.Ops.conv2d(l9, k=3, out_c=en_c[4], str=1, name='en_c10'), label='lrelu')
			print("l10_r2n",l10.shape)#l10_r2n (?, 8, 8, 256)
			l88 = tools.Ops.conv2d(l8, k=1, out_c=en_c[4], str=1, name='en_c88')
			print("l88_r2n",l88.shape) #l88_r2n (?, 8, 8, 256)
			l10 = l10 + l88
			l10 = tools.Ops.maxpool2d(l10, k=2, s=2, name='en_mp5')
			print("l10_r2n",l10.shape) #l10_r2n (?, 4, 4, 256)
			
#			plot_buf_1= tf.reshape(l10, [-1, 4, 4, 3]) ########################
#			tf.summary.image("l10_en", plot_buf_1)#############################

			l11 = tools.Ops.xxlu(tools.Ops.conv2d(l10, k=3, out_c=en_c[5], str=1, name='en_c11'), label='lrelu')
			print("l11_r2n",l11.shape) #l11_r2n (?, 4, 4, 256)
			l12 = tools.Ops.xxlu(tools.Ops.conv2d(l11, k=3, out_c=en_c[5], str=1, name='en_c12'), label='lrelu')
			print("l12_r2n",l12.shape) #l12_r2n (?, 4, 4, 256)
			l1010 = tools.Ops.conv2d(l10, k=1, out_c=en_c[5], str=1, name='en_c1010')
			print("l1010_r2n",l1010.shape) #l1010_r2n (?, 4, 4, 256)
			l12 = l12 + l1010
			l12 = tools.Ops.maxpool2d(l12, k=2, s=2, name='en_mp6')
			print("l12_r2n",l12.shape) #l12_r2n (?, 2, 2, 256)
			
#			plot_buf_1= tf.reshape(l12, [-1, 2, 2, 3]) ########################
#			tf.summary.image("l12_en", plot_buf_1)#############################
		
			[_, d1, d2, cc] = l12.get_shape()
			l12 = tf.reshape(l12, [-1, int(d1) * int(d2) * int(cc)],name="en_fc1_in")
			print("fc1_input_r2n",l12.shape) #fc1_input_r2n (?, 1024)
			fc = tools.Ops.xxlu(tools.Ops.fc(l12, out_d=1024, name='en_fc1'), label='lrelu')
			print("fc1_output_r2n",fc.shape)#fc1_output_r2n (?, 1024)
			
		with tf.variable_scope('Att_Net'):	
			#### use fc attention
			input = tf.reshape(fc, [-1, im_num, 1024],name="Att_fc_in")
			print("att_fc_in_r2n",input.shape) #att_fc_in_r2n (?, ?, 1024)
			latent_3d, weights = attsets_fc(input, out_ele_num=1)
			print("att_fc_out_r2n",latent_3d.shape) #att_fc_out_r2n (?, 1024)
			
		with tf.variable_scope('Decoder'):
			####
			latent_3d = tools.Ops.xxlu(tools.Ops.fc(latent_3d, out_d=4*4*4*128, name='de_fc2'), label='lrelu')
			print("fc3_out_r2n",latent_3d.shape) #fc3_out_r2n (?, 8192)
			latent_3d = tf.reshape(latent_3d, [-1, 4, 4, 4, 128],name="de_fc2_out")

		####

			de_c = [128, 128, 128, 64, 32, 1]
			
			print("d1_in_r2n",latent_3d.shape) #d1_in_r2n (?, 4, 4, 4, 128)
			d1 = tools.Ops.xxlu(tools.Ops.deconv3d(latent_3d, k=3, out_c=de_c[1], str=2, name='de_c1'), label='lrelu')
			print("d1_out_r2n",d1.shape) #d1_out_r2n (?, 8, 8, 8, 128)
			d2 = tools.Ops.xxlu(tools.Ops.deconv3d(d1, k=3, out_c=de_c[1], str=1, name='de_c2'), label='lrelu')
			print("d2_out_r2n",d2.shape) #d2_out_r2n (?, 8, 8, 8, 128)
			d00 = tools.Ops.deconv3d(latent_3d, k=1, out_c=de_c[1], str=2, name='de_c00')
			print("d00_out_r2n",d00.shape)#d00_out_r2n (?, 8, 8, 8, 128)
			d2 = d2 + d00
			print("d2+d00_out_r2n",d2.shape)#d2+d00_out_r2n (?, 8, 8, 8, 128)
			
#			plot_buf_1= tf.reshape(d2, [-1, 8, 8, 4]) ################################
#			tf.summary.image("d2+d00_out_de", plot_buf_1)#############################

			d3 = tools.Ops.xxlu(tools.Ops.deconv3d(d2, k=3, out_c=de_c[2], str=2, name='de_c3'), label='lrelu')
			print("d3_out_r2n",d3.shape)#d3_out_r2n (?, 16, 16, 16, 128)
			d4 = tools.Ops.xxlu(tools.Ops.deconv3d(d3, k=3, out_c=de_c[2], str=1, name='de_c4'), label='lrelu')
			print("d4_out_r2n",d4.shape)#d4_out_r2n (?, 16, 16, 16, 128)
			d22 = tools.Ops.deconv3d(d2, k=1, out_c=de_c[2], str=2, name='de_c22')
			print("d22_out_r2n",d22.shape)#d22_out_r2n (?, 16, 16, 16, 128)
			d4 = d4 + d22
			print("d4+d22_out_r2n",d4.shape)#d4+d22_out_r2n (?, 16, 16, 16, 128)
			
#			plot_buf_1= tf.reshape(d4, [-1, 16, 16, 4]) ##############################
#			tf.summary.image("d4+d22_out_de", plot_buf_1)#############################

			d5 = tools.Ops.xxlu(tools.Ops.deconv3d(d4, k=3, out_c=de_c[3], str=2, name='de_c5'), label='lrelu')
			print("d5_out_r2n",d5.shape)#d5_out_r2n (?, 32, 32, 32, 64)
			d6 = tools.Ops.xxlu(tools.Ops.deconv3d(d5, k=3, out_c=de_c[3], str=1, name='de_c6'), label='lrelu')
			print("d6_out_r2n",d6.shape)#d6_out_r2n (?, 32, 32, 32, 64)
			d44 = tools.Ops.deconv3d(d4, k=1, out_c=de_c[3], str=2, name='de_c44')
			print("d44_out_r2n",d44.shape)#d44_out_r2n (?, 32, 32, 32, 64)
			d6 = d6 + d44
			print("d6+d44_out_r2n",d6.shape) #d6+d44_out_r2n (?, 32, 32, 32, 64)
			
#			plot_buf_1= tf.reshape(d6, [-1, 32, 32, 4]) ##############################
#			tf.summary.image("d6+d44_out_de", plot_buf_1)#############################

			d7 = tools.Ops.xxlu(tools.Ops.deconv3d(d6, k=3, out_c=de_c[4], str=1, name='de_c7'), label='lrelu')
			print("d7_out_r2n",d7.shape) #d7_out_r2n (?, 32, 32, 32, 32)
			d8 = tools.Ops.xxlu(tools.Ops.deconv3d(d7, k=3, out_c=de_c[4], str=1, name='de_c8'), label='lrelu')
			print("d8_out_r2n",d8.shape)#d8_out_r2n (?, 32, 32, 32, 32)
			d77 = tools.Ops.xxlu(tools.Ops.deconv3d(d7, k=3, out_c=de_c[4], str=1, name='de_c77'), label='lrelu')
			print("d77_out_r2n",d77.shape)#d77_out_r2n (?, 32, 32, 32, 32)
			d8 = d8 + d77
			print("d8+d77_out_r2n",d8.shape) #d8+d77_out_r2n (?, 32, 32, 32, 32)
			
#			plot_buf_1= tf.reshape(d8, [-1, 32, 32, 4]) ##############################
#			tf.summary.image("d8+d77_out_de", plot_buf_1)#############################

			d11 = tools.Ops.deconv3d(d8, k=3, out_c=de_c[5], str=1, name='de_c9')
			print("d11_out_r2n",d11.shape) #d11_out_r2n (?, 32, 32, 32, 1)
			
#			plot_buf_1= tf.reshape(d11, [-1, 32, 32,4]) ##########################
#			tf.summary.image("Ref_input", plot_buf_1)#############################
			
			ref_in = tf.reshape(d11, [-1, vox_res32, vox_res32, vox_res32],name="ref_in")     ###
			
			y = tf.nn.sigmoid(d11,name='de_sigmoid')

			att_o = tf.reshape(y, [-1, vox_res32, vox_res32, vox_res32],name="de_out")
			
			print("att_out_shape",att_o.shape) #att_out_shape (?, 32, 32, 32)
			
		with tf.variable_scope('ref_net'):
		
			ref_o=refiner_network(ref_in)
			
#			plot_buf_1= tf.reshape(ref_o, [-1, 32, 32,4]) ######################
#			tf.summary.image("Ref_Out", plot_buf_1)#############################
			
			return ref_o,att_o, weights

	def build_graph(self):
		img_res = 127
		vox_res = 32
		self.X_rgb = tf.placeholder(shape=[None, None, img_res, img_res, 3], dtype=tf.float32)
		self.Y_vox = tf.placeholder(shape=[None, vox_res, vox_res, vox_res], dtype=tf.float32)
		self.lr = tf.placeholder(tf.float32)
		self.refine_lr = tf.placeholder(tf.float32)
		with tf.device('/gpu:' + GPU0):
			self.Y_pred,self.vae_o, self.weights = self.base_r2n2(self.X_rgb)
			tf.summary.histogram('Attsets_Weights', self.weights)
		with tf.device('/gpu:' + GPU1):	
			### rec loss
			print ("reached")
			with tf.variable_scope('Loss_Fun'):
				Y_vox_ = tf.reshape(self.Y_vox, shape=[-1, vox_res ** 3])
				Y_pred_ = tf.reshape(self.Y_pred, shape=[-1, vox_res ** 3])
				vae_o_=tf.reshape(self.vae_o, shape=[-1, vox_res ** 3])
			
				self.vae_loss = tf.reduce_mean(-tf.reduce_mean(Y_vox_ * tf.log(vae_o_ + 1e-8), reduction_indices=[1]) -
										 tf.reduce_mean((1 - Y_vox_) * tf.log(1 - vae_o_ + 1e-8),reduction_indices=[1]))
				self.rec_loss = tf.reduce_mean(-tf.reduce_mean(Y_vox_ * tf.log(Y_pred_ + 1e-8), reduction_indices=[1]) -
										 tf.reduce_mean((1 - Y_vox_) * tf.log(1 - Y_pred_ + 1e-8),reduction_indices=[1]))
				sum_rec_loss = tf.summary.scalar('rec_loss', self.rec_loss)
				self.sum_merged = sum_rec_loss
				tf.summary.histogram('rec_loss', self.rec_loss)
				tf.summary.scalar("vae-loss",self.vae_loss)
				tf.summary.histogram("vae_loss",self.vae_loss)
				self.mean_loss = tf.div(x=tf.math.add(x=self.vae_loss,y=self.rec_loss,name='add_loss'),y=2,name='mean_loss')
				tf.summary.histogram("mean_vae_loss",self.mean_loss)
				tf.summary.scalar("mean_vae_loss",self.mean_loss)
				
			with tf.variable_scope('Evaluation_Metric'):			    
                		gt_vox=Y_vox_
                		self.iou_ref = metric_iou(Y_pred_,gt_vox)
                		tf.summary.scalar('iou_refiner', self.iou_ref)
                		tf.summary.histogram('iou_refiner', self.iou_ref)
                		self.iou_vae = metric_iou(vae_o_,gt_vox)
                		tf.summary.scalar('iou_vae', self.iou_vae)
                		tf.summary.histogram("iou_vae",self.iou_vae)
                		
			with tf.variable_scope('Optimization'):

                		base_en_var = [var for var in tf.trainable_variables() if var.name.startswith('Encoder/en')]
                		base_dec_var = [var for var in tf.trainable_variables() if var.name.startswith('Decoder/de')]
                		att_var = [var for var in tf.trainable_variables() if var.name.startswith('Att_Net/att')]
                		refine_var = [var for var in tf.trainable_variables() if var.name.startswith('ref_net/ref')]
                		self.refine_optim = tf.train.AdamOptimizer(learning_rate=self.refine_lr).minimize(self.rec_loss, var_list=refine_var)
                		self.base_en_optim2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.vae_loss, var_list=base_en_var)
                		self.base_de_optim2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.vae_loss, var_list=base_dec_var)
                		self.att_optim2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.vae_loss, var_list=att_var)
			    				
		
		print ("total weights:",tools.Ops.variable_count())
		self.saver = tf.train.Saver(max_to_keep=1)
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		config.gpu_options.visible_device_list = '0,1'
		self.sess = tf.Session(config=config)
		self.merged = tf.summary.merge_all()
		self.sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, self.sess.graph)
		self.sum_writer_test = tf.summary.FileWriter(self.test_sum_dir, self.sess.graph)

		#######################
		path = self.train_mod_dir
		#path = './Model_released/'  # retrain the released model

		if os.path.isfile(path + 'model.cptk.data-00000-of-00001'):
			print ("restoring saved model!")
			self.saver.restore(self.sess, path + 'model.cptk')
		else:
			self.sess.run(tf.global_variables_initializer())
		return 0
    
	def train(self, data):
		for epoch in range(0, 500, 1):
			train_view_num = 24  ##!!!!!!!!!!!
			data.shuffle_train_files(epoch, train_mv=train_view_num)
			total_train_batch_num = data.total_train_batch_num  #int(len(self.X_rgb_train_files)/(self.batch_size*train_mv))
			print ('total_train_batch_num:', total_train_batch_num)
			for i in range(total_train_batch_num):
				#### training
				X_rgb_bat, Y_vox_bat = data.load_X_Y_train_next_batch(train_mv=train_view_num)
				print("multi_view_train_X_rgb_bat : ",X_rgb_bat.shape)#np.asarray(X.append(X_rgb[b*train_mv:(b+1)*train_mv,:]))
				

				print(time.ctime())
				
				##### option 1: seperate train, seperate optimize
				#if epoch<=30:
				#	single_view_train=True
				#	multi_view_train=False
				#else:
				#	single_view_train=False
				#	multi_view_train=True

				##### optiion 2: joint train, seperate optimize
				single_view_train = True
				multi_view_train = False
				
				if epoch <= 5:
					att_lr=.0002
					ref_lr=.0002
				if epoch > 5 and epoch <= 50:
					att_lr=.0001
					ref_lr=.0001
				if epoch > 50:
					att_lr=.00005
					ref_lr=.00005 
				###########  single view train
				if single_view_train:
					
					rgb = np.reshape(X_rgb_bat,[batch_size*train_view_num, 1, 127,127,3])
					print("single_view_train_rgb_input_shape ",rgb.shape)
					vox = np.tile(Y_vox_bat[:,None,:,:,:],[1,train_view_num,1,1,1])
					vox = np.reshape(vox, [batch_size*train_view_num, 32,32,32])	
					vae_loss_c,eee,ddd, rec_loss_c, sum_train,rrr,mean_vae,iou_ref_,iou_vae_ = self.sess.run([self.vae_loss,self.base_en_optim2,self.base_de_optim2,self.rec_loss,self.merged,self.refine_optim,self.mean_loss,self.iou_ref,self.iou_vae],feed_dict={self.X_rgb: rgb, self.Y_vox: vox, self.lr: att_lr,self.refine_lr: ref_lr})
					print ('ep:', epoch, 'i:', i, 'train single rec loss:', rec_loss_c)
					print ('ep:', epoch, 'i:', i, 'train single vae loss:', vae_loss_c)
					print ('ep:', epoch, 'i:', i, 'train single mean_vae loss:',mean_vae)
					print ('ep:', epoch, 'i:', i, 'train single ref_iou:',iou_ref_)
					print ('ep:', epoch, 'i:', i, 'train single vae_iou:',iou_vae_)
                                        									
				########## multi view train
				if multi_view_train:
					
					vae_loss_c,rec_loss_c, _, sum_train,xxx,mean_vae,iou_ref_,iou_vae_ = self.sess.run([self.vae_loss,self.rec_loss, self.att_optim2, self.merged,self.refine_optim,self.mean_loss,self.iou_ref,self.iou_vae],feed_dict={self.X_rgb: X_rgb_bat, self.Y_vox: Y_vox_bat,self.lr: att_lr,self.refine_lr: ref_lr})
					print ('ep:', epoch, 'i:', i, 'train multi rec loss:', rec_loss_c)
					print ('ep:', epoch, 'i:', i, 'train multi vae loss:', vae_loss_c)
					print('ep:',epoch,'i',i,'train multi mean_vae loss:',mean_vae)
					print ('ep:', epoch, 'i:', i, 'train multi ref_iou:',iou_ref_)
					print ('ep:', epoch, 'i:', i, 'train multi vae_iou:',iou_vae_)
                                        				
				############
				if i % 10 == 0:
					self.sum_writer_train.add_summary(sum_train, epoch * total_train_batch_num + i)
					
				
				#### testing
				if i % 50 == 0 :
					X_rgb_batch, Y_vox_batch = data.load_X_Y_test_next_batch(test_mv=3)
					
#					vae_pred = tf.get_default_graph().get_tensor_by_name("Decoder/de_out:0")
#					ref_pred = tf.get_default_graph().get_tensor_by_name("ref_net/ref_Dec/ref_out:0")
#					gt_vox=Y_vox_batch.astype(np.float32)
					
#					iou_pred = metric_iou(ref_pred,gt_vox)
#					tf.summary.scalar("iou",iou_pred)

					rrrr,aaaa,rec_loss_te, qwerty, Y_vox_test_pred, att_pred, sum_test,mean_vae,iou_ref_,iou_vae_ = \
						self.sess.run([self.refine_optim,self.att_optim2,self.rec_loss,self.vae_loss, self.Y_pred,self.weights, self.merged,self.mean_loss,self.iou_ref,self.iou_vae],feed_dict={self.X_rgb: X_rgb_batch, self.Y_vox: Y_vox_batch,self.lr: att_lr,self.refine_lr: ref_lr})
						
					X_rgb_batch = X_rgb_batch.astype(np.float16)
					
					Y_vox_batch = Y_vox_batch.astype(np.float16)
					Y_vox_test_pred = Y_vox_test_pred.astype(np.float16)
					att_pred = att_pred.astype(np.float16)
					to_save = {'X_test':X_rgb_batch,'Y_test_pred':Y_vox_test_pred,'att_pred':att_pred,'Y_test_true':Y_vox_batch}
					scipy.io.savemat(self.test_res_dir+'X_Y_pred_'+str(epoch).zfill(2)+'_'+str(i).zfill(5)+'.mat',to_save,do_compression=True)
					
					
					self.sum_writer_test.add_summary(sum_test, epoch * total_train_batch_num + i)
										
#					iou_ref=evaluate_voxel_prediction(ref_pred_,gt_vox)
#					iou_vae=evaluate_voxel_prediction(vae_pred_,gt_vox)
					
#					print("Ref_iou:",iou_ref)
#					print("Vae_iou:",iou_vae)
					
#					plot_list_iou.append(iou_ref)
#					plot_list_i.append((i/50))
#					graph_plot(plot_list_iou,plot_list_i)
					print ('ep:', epoch, 'i:', i, 'test rec loss:', rec_loss_te)
					print ('ep:', epoch, 'i:', i, 'test vae loss:', qwerty )
					print ('ep:', epoch, 'i:', i, 'test mean_vae loss:', mean_vae) 
					print ('ep:', epoch, 'i:', i, 'test ref_iou:',iou_ref_)
					print ('ep:', epoch, 'i:', i, 'test vae_iou:',iou_vae_)
					
				#### model saving
				if i % 100 == 0 :
					self.saver.save( self.sess, save_path=self.train_mod_dir + 'model.cptk' )
					print ( 'epoch:', epoch, 'i:', i, 'model saved!' )
#					plt.show()
				

									

##########
if __name__ =='__main__':

		net = Network()          #net=object to create instance

		print("network compleated")   ###

		net.build_graph()
		print("graph compleated")
                
#               sys.exit(). sys.exit()        ###
		
		data = tools.Data(config)
		print("tools.data compleated")

                
		print('training data')
		
		net.train(data)


			
           


	

	


