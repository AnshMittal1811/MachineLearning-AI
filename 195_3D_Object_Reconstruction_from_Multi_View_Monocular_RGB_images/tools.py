import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
import re
import binvox_rw as binvox_rw
#################################################################
from voxel import voxel2obj
import mcubes
from export_obj import export_obj 
#################################################################
class Data:
    def __init__(self, config):
        self.config = config
        self.batch_size = config['batch_size']
        self.train_batch_index = 0
        self.test_batch_index_sq = 0
        self.cat_names = config['cat_names']
        self.total_mv = config['total_mv']
        self.cat_test_1st_index = None
        
        self.X_rgb_train_files_ori, self.Y_vox_train_files_ori, self.X_rgb_test_files_ori, self.Y_vox_test_files_ori \
            =self.load_X_Y_files_paths_all(self.cat_names)

        print ('X_rgb_train_files_ori:', len(self.X_rgb_train_files_ori))
        print ('X_rgb_test_files_ori:',len(self.X_rgb_test_files_ori))

    @staticmethod
    def plotFromVoxels(voxels,title=''):
#        print('plotfromvoxel')
        voxel2obj(title+'voxels.obj', voxels)
        if len(voxels.shape) > 3:
            x_d = voxels.shape[0]
            y_d = voxels.shape[1]
            z_d = voxels.shape[2]
            v = voxels[:, :, :, 0]
           
            v = np.reshape(v, (x_d, y_d, z_d))
        else:
            v = voxels
        
        print("voxels_plot",v.shape)   ###############################################  (32, 32, 32)  ##################################
         		
        u=voxels
        vertices, triangles = mcubes.marching_cubes(u, 0)
        mcubes.export_mesh(vertices, triangles, title+"recon.dae", "MySphere")
        export_obj(vertices, triangles,title+'recon.obj')

        x, y, z = v.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('equal')
        ax.view_init(-90, 90)

        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
        print("max_range",max_range)

        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        plt.grid()

       	plt.show()
        plt.title(title)
        from matplotlib.pyplot import show
        show(block=False)
        

    ########## from 3D-R2N2
    @staticmethod
    def crop_center(im, new_height, new_width):
#        print('crop_center')
        height = im.shape[0]  # Get dimensions
        width = im.shape[1]
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = (width + new_width) // 2
        bottom = (height + new_height) // 2
        return im[top:bottom, left:right]
    
    @staticmethod
    def add_random_color_background(im, color_range):
#        print('add_random_color_background')
        r, g, b = [np.random.randint(color_range[i][0], color_range[i][1] + 1) for i in range(3)]
#        print (r)
#        print (g)
#        print (b)
        if isinstance(im, Image.Image):
            im = np.array(im) 
#        print (im.shape)
#        b = np.array([3])         ###
#        np.concatenate((im, '3'))    ##
#        im=np.append(im,3)        ###
#        print (im.shape)         ###
        if im.shape[2] > 3:
              
            # If the image has the alpha channel, add the background
            alpha = (np.expand_dims(im[:, :, 3], axis=2) == 0).astype(np.float)
	   
            im = im[:, :, :3]
            bg_color = np.array([[[r, g, b]]])
            im = alpha * bg_color + (1 - alpha) * im
        return im
    
    @staticmethod
    def image_transform(img, crop_x, crop_y, crop_loc=None, color_tint=None):
#        print('image_transform')
        RANDOM_CROP = True
        # Slight translation
        if RANDOM_CROP and not crop_loc:
            crop_loc = [np.random.randint(0, crop_y), np.random.randint(0, crop_x)]
    
        if crop_loc:
            cr, cc = crop_loc
            height, width, _ = img.shape
            img_h = height - crop_y
            img_w = width - crop_x
            img = img[cr:cr + img_h, cc:cc + img_w]
            # depth = depth[cr:cr+img_h, cc:cc+img_w]

        FLIP = True
        if FLIP and np.random.rand() > 0.5:
            img = img[:, ::-1, ...]
        return img
    
    @staticmethod
    def preprocess_img(im, train):
#        print('processing_img')
        # add random background
        TRAIN_NO_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]
        TEST_NO_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]
        im = Data.add_random_color_background(im, TRAIN_NO_BG_COLOR_RANGE if train else TEST_NO_BG_COLOR_RANGE)
    
        # If the image has alpha channel, remove it.
        CONST_IMG_W = 127
        CONST_IMG_H = 127
        im_rgb = np.array(im)[:, :, :3].astype(np.float32)
        if train:
            # Data augmentation
            PAD_X = 10
            PAD_Y = 10
            t_im = Data.image_transform(im_rgb, PAD_X, PAD_Y)
        else:
            t_im = Data.crop_center(im_rgb, CONST_IMG_H, CONST_IMG_W)
    
        # Scale image
        t_im = t_im / 255.
        return t_im
    
    @staticmethod
    def load_single_X_rgb_r2n2(img_path, train):
#        print('load_single_X_rgb_r2n2')
        im = Image.open(img_path)
        
        t_im = Data.preprocess_img(im, train=train)
       # plt.figure()
       # plt.imshow(t_im)
        return t_im

    @staticmethod
    def load_single_Y_vox(vox_path):
#        print('load_single_Y_vox')
        with open(vox_path, 'rb') as ff:
            vox = binvox_rw.read_as_3d_array(ff)
            vox_grid = vox.data.astype(int)

        #Data.plotFromVoxels(vox_grid)
        return vox_grid

    @staticmethod
    def load_X_Y_files_paths(X_cat_folder, Y_cat_folder):
#        print('load_X_Y_files_paths')
        X_obj_folders=[X_f for X_f in sorted(os.listdir(X_cat_folder))]
        Y_obj_folders=[Y_f for Y_f in sorted(os.listdir(Y_cat_folder))]
        if len(X_obj_folders) != len(Y_obj_folders):
            print ('Files are inconsistent in:', X_cat_folder, 'and', Y_cat_folder)

        #### split train/test, according to 3D-R2N2 paper
        train_num = int(0.8*len(X_obj_folders))
        idx = list(range(train_num))
        
        test_objs = []
        X_train_obj_folders =[];  Y_train_obj_folders=[]
        X_test_obj_folders =[]; Y_test_obj_folders=[]
        for i in range(len(X_obj_folders)):
            obj_na = X_obj_folders[i]
            if obj_na not in Y_obj_folders:
                print ('inconsistent single obj ignored')
                continue
            if i in idx:
                X_train_obj_folders.append(X_cat_folder+obj_na+'/')
                Y_train_obj_folders.append(Y_cat_folder+obj_na+'/')
            else:
                X_test_obj_folders.append(X_cat_folder+obj_na+'/')
                Y_test_obj_folders.append(Y_cat_folder+obj_na+'/')
                test_objs.append(obj_na)
        print ('train objs:', len(X_train_obj_folders))
        print ('test objs:', len(X_test_obj_folders))

        #########
        def load_x_y_files(X_obj_fo, Y_obj_fo):
#            print('load_x_y_files')
            X_files_paths = []
            Y_files_paths = []
            for j in range(len(X_obj_fo)):
                if X_obj_fo[j][-5:] not in Y_obj_fo[j]:
                    print ('inconsistent single obj exit')
                    exit()
                for xf in sorted(os.listdir(X_obj_fo[j]+'rendering/')):
                    if '.png' in xf:
                        X_files_paths.append(X_obj_fo[j]+'rendering/'+xf)
                        Y_files_paths.append(Y_obj_fo[j]+'model.binvox')
            return X_files_paths, Y_files_paths
        #########

        X_train_files, Y_train_files = load_x_y_files(X_train_obj_folders, Y_train_obj_folders)
        X_test_files, Y_test_files = load_x_y_files(X_test_obj_folders, Y_test_obj_folders)

        return X_train_files, Y_train_files,X_test_files,Y_test_files

    @staticmethod
    def load_X_Y_rgb_vox(X_files_full_path, Y_files_full_path, train):
#        print('load_X_Y_rgb_vox')
        if len(X_files_full_path) != len(Y_files_full_path):
            print ('load_X_Y_rgb_vox error!')
            exit()
        X_rgb = []
        Y_vox = []
        for X_f, Y_f in zip(X_files_full_path, Y_files_full_path):
            na = re.split('/', X_f)[-3]
            if na not in Y_f:
                print ('X Y rgb vox file not consistent!')
                exit()
            rgb = Data.load_single_X_rgb_r2n2(X_f, train=train)
            X_rgb.append(rgb)

            vox = Data.load_single_Y_vox(Y_f)
            Y_vox.append(vox)

        X_rgb = np.asarray(X_rgb)
        Y_vox = np.asarray(Y_vox)
        return X_rgb, Y_vox

    def load_X_Y_files_paths_all(self,cat_names):
#        print('load_X_Y_files_paths_all')
        x_rgb_str='X_rgb_'
        y_vox_str='Y_vox_'

        X_train_files_paths_all=[]
        Y_trian_files_paths_all=[]
        X_test_files_paths_all=[]
        Y_test_files_paths_all=[]
        self.cat_test_1st_index = [0]
        for name in cat_names:
            print ('loading files:', name)
            X_rgb_folder = self.config[x_rgb_str+name]
            Y_vox_folder = self.config[y_vox_str+name]

            X_train_files, Y_train_files, X_test_files, Y_test_files = self.load_X_Y_files_paths(X_rgb_folder, Y_vox_folder)
            self.cat_test_1st_index.append(len(X_test_files))

            for X_rgb_f, Y_vox_f in zip(X_train_files, Y_train_files):
                X_train_files_paths_all.append(X_rgb_f)
                Y_trian_files_paths_all.append(Y_vox_f)
            for X_rgb_f, Y_vox_f in zip(X_test_files, Y_test_files):
                X_test_files_paths_all.append(X_rgb_f)
                Y_test_files_paths_all.append(Y_vox_f)

        return X_train_files_paths_all, Y_trian_files_paths_all,X_test_files_paths_all,Y_test_files_paths_all

    ################################
    def load_X_Y_train_next_batch(self, train_mv):
#        print('load_X_Y_train_next_batch')
        X_rgb_files = self.X_rgb_train_files[self.batch_size*self.train_batch_index*train_mv:self.batch_size*(self.train_batch_index+1)*train_mv]
        Y_vox_files = self.Y_vox_train_files[self.batch_size*self.train_batch_index*train_mv:self.batch_size*(self.train_batch_index+1)*train_mv]
        self.train_batch_index +=1
        
        X_rgb, Y_vox = self.load_X_Y_rgb_vox(X_rgb_files, Y_vox_files, train=True)
        X = []
        Y = []
        for b in range(self.batch_size):
            X.append(X_rgb[b * train_mv:(b + 1) * train_mv, :])
            Y.append(Y_vox[b * train_mv, :, :, :])
        X = np.asarray(X)
        Y = np.asarray(Y)
        return X, Y

    def load_X_Y_test_next_batch(self, test_mv):
#        print('load_X_Y_test_next_batch')
        num = self.total_mv
        idx = random.sample(range(len(self.X_rgb_test_files_ori)//num), self.batch_size)  ##############added extra '/'
        X_rgb_files = []
        Y_vox_files =[]

        for i in idx:
            tp1 = self.X_rgb_test_files_ori[i*num:(i+1)*num]
            tp2 = self.Y_vox_test_files_ori[i*num:(i+1)*num]
            for x, y in zip(tp1, tp2):
                X_rgb_files.append(x)
                Y_vox_files.append(y)
        X_rgb_batch, Y_vox_batch = self.load_X_Y_rgb_vox(X_rgb_files, Y_vox_files, train=False)
        
        X =[]
        Y =[]
        for b in range(self.batch_size):
            tp1 = X_rgb_batch[b*num:(b+1)*num,:,:,:]
            tp2 = Y_vox_batch[b*num:(b+1)*num,:,:,:]
            idx2 = random.sample(range(num), test_mv)
            X.append(tp1[idx2])
            Y.append(tp2[idx2[0]])
        X = np.asarray(X)
        Y = np.asarray(Y)
        return X, Y

    def shuffle_train_files(self, ep, train_mv):
#        print('shuffle_train_files')
        num = self.total_mv
        X_rgb_new=[]; Y_vox_new=[]
        self.train_batch_index = 0
        X_rgb = self.X_rgb_train_files_ori
        Y_vox = self.Y_vox_train_files_ori
        index = list(range(int(len(X_rgb)/num)))
        random.Random(ep).shuffle(index)

        for i in index:
            tp1 = X_rgb[i*num:(i+1)*num]
            tp2 = Y_vox[i*num:(i+1)*num]
            
            view_ind = list(range(num))
            random.Random(i+100).shuffle(view_ind)
            valid_view_num = int(num / train_mv) * train_mv
            view_ind = view_ind[0:valid_view_num]
            
            for j in view_ind:
                x = tp1[j]
                y = tp2[j]
                na = re.split('/', x)[-3]
                if na not in y:
                    print ('X Y rgb vox file not consistent!')
                    exit()
                X_rgb_new.append(x)
                Y_vox_new.append(y)
        self.X_rgb_train_files = X_rgb_new
        self.Y_vox_train_files = Y_vox_new
        self.total_train_batch_num = int(len(self.X_rgb_train_files)/(self.batch_size*train_mv))
    
    def shuffle_test_files(self, test_mv, seed):
#        print('shuffle_test_files')
        X_rgb_new=[]; Y_vox_new=[]
        X_rgb = self.X_rgb_test_files_ori
        Y_vox = self.Y_vox_test_files_ori

        num = self.total_mv
        ###
        cat_test_1st_index_new =[]
        for k in self.cat_test_1st_index:
            cat_test_1st_index_new.append( int(k/num)* int(num/test_mv)*test_mv)
        self.cat_test_1st_index=cat_test_1st_index_new
        ###
        total_obj = int(len(X_rgb)/num)
        for i in range(total_obj):
            view_ind = list(range(num))
            random.Random(i+seed).shuffle(view_ind)
            
            valid_view_num = int(num/test_mv)*test_mv
            view_ind = view_ind[0:valid_view_num]
            for id in view_ind:
                X_rgb_new.append(X_rgb[i*num+id])
                Y_vox_new.append(Y_vox[i*num+id])
        
        self.X_rgb_test_files = X_rgb_new
        self.Y_vox_test_files = Y_vox_new
        
class Ops:

    @staticmethod
    def lrelu(x, leak=0.2):
       return  tf.nn.leaky_relu(x,alpha=0.2,name=None)
	   
    @staticmethod
    def relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def xxlu(x, label='relu'):
        if label == 'relu':
            return Ops.relu(x)
        if label == 'lrelu':
            return Ops.lrelu(x, leak=0.2)

    @staticmethod
    def variable_sum(var, name):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def variable_count():
        total_para = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        return total_para

    @staticmethod
    def fc(x, out_d, name):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_d = x.get_shape()[1]
        w = tf.get_variable(name + '_w', [in_d, out_d], initializer=xavier_init,dtype=tf.float32)
        b = tf.get_variable(name + '_b', [out_d], initializer=zero_init,dtype=tf.float32)
        y = tf.nn.bias_add(tf.matmul(x, w), b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def conv2d(x, k, out_c, str, name, pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_c = x.get_shape()[3]
        w = tf.get_variable(name + '_w', [k, k, in_c, out_c], initializer=xavier_init, dtype=tf.float32)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init,dtype=tf.float32)

        stride = [1, str, str, 1]
        y = tf.nn.bias_add(tf.nn.conv2d(x, w, stride, pad), b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def maxpool2d(x, k, s,name, pad='SAME'):
        ker = [1,k,k,1]
        str = [1,s,s,1]
        y = tf.nn.max_pool(x,ksize=ker, strides=str, padding=pad, name=name)
        return y

    @staticmethod
    def conv3d(x, k, out_c, str, name, pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_c = x.get_shape()[4]
        w = tf.get_variable(name + '_w', [k, k, k, in_c, out_c], initializer=xavier_init, dtype=tf.float32)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init, dtype=tf.float32)
        stride = [1, str, str, str, 1]
        y = tf.nn.bias_add(tf.nn.conv3d(x, w, stride, pad), b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def deconv3d(x, k, out_c, str, name, pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        bat = tf.shape(x)[0]
        [_, in_d1, in_d2, in_d3, in_c] = x.get_shape()
        in_d1 = int(in_d1); in_d2 = int(in_d2); in_d3 = int(in_d3); in_c = int(in_c)
        w = tf.get_variable(name + '_w', [k, k, k, out_c, in_c], initializer=xavier_init, dtype=tf.float32)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init, dtype=tf.float32)

        out_shape = [bat, in_d1 * str, in_d2 * str, in_d3 * str, out_c]
        stride = [1, str, str, str, 1]
        y = tf.nn.conv3d_transpose(x, w, output_shape=out_shape, strides=stride, padding=pad)
        y = tf.nn.bias_add(y, b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def maxpool3d(x,k,s,name,pad='SAME'):
        ker = [1,k,k,k,1]
        str = [1,s,s,s,1]
        y = tf.nn.max_pool3d(x,ksize=ker,strides=str,padding=pad, name=name)
        return y


        
