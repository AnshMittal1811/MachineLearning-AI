
#====================================================================================================================================================

# 14 category of objects    
# scene size: 84 x 44 x 84  
# give gt 2d as input   

#====================================================================================================================================================

import sys, glob, datetime, time, random, os, os.path, shutil, logging
import numpy            as     np
from   random           import randint
from   numpy            import array 
from   collections      import Counter
from   multiprocessing  import Pool 
import tensorflow       as     tf 
import utils # TODO fix it later

#====================================================================================================================================================

classes_count        = 14
scene_shape          = [84, 44, 84]
halfed_scene_shape   = scene_shape[2] / 2  
directory            = 'hybrid_model_v2'
to_train             = True
to_restore           = False
show_accuracy        = True
show_accuracy_step   = 500
save_model           = True
save_model_step      = 5000
visualize_scene      = True
visualize_scene_step = 5000
subset_train         = False 
train_directory      = 'house_2/' 
test_directory       = 'test_data_2/'
train_2d_directory   = 'house_2d/' 
test_2d_directory    = 'house_2d/'
max_iter             = 500000
learning_rate        = 0.00001
batch_size           = 16  
num_of_vis_batch     = 1
cardinality          = 8 # how many split  
blocks               = 3 # res_block (split + transition)

logging.basicConfig(filename=str(directory)+'.log', level=logging.DEBUG) 

if not os.path.exists(directory):
    os.makedirs(directory)
    
train_data    = []  
train_data_2d = []  
test_data     = []
test_data_2d  = []

for item in glob.glob(train_directory + "*.npy"):
    train_data.append(item)
    
for item in glob.glob(train_2d_directory + "*.npy"):
    train_data_2d.append(item)
    
for item in glob.glob(test_directory + "*.npy"):
    test_data.append(item)
    
for item in glob.glob(test_2d_directory + "*.npy"):
    test_data_2d.append(item)

batch_threshold = 0
if subset_train:
    batch_threshold = batch_size * visualize_scene_step
else:
    batch_threshold = len(train_data)
    
#=====================================================================================================================================================

def count_params():
    "print number of trainable variables"
    size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
    n = sum(size(v) for v in tf.trainable_variables())
    print "Model size: %dK" % (n/1000,)

#=====================================================================================================================================================

class ConvNet(object):

    def paramsFun(self): 
        params_cnn_w = {
                    'w1'   : tf.Variable(tf.truncated_normal( [ 7 , 7 , halfed_scene_shape , 42               ], stddev = 0.01 )),   
                    'w2'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42                               ], stddev = 0.01 )),  
                    'w3'   : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42                               ], stddev = 0.01 )),
                    'w4'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42                               ], stddev = 0.01 )),   
                    'w5'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42                               ], stddev = 0.01 )),  
                    'w6'   : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42                               ], stddev = 0.01 )),  
                    'w7'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42                               ], stddev = 0.01 )),   
                    'w8'   : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42                               ], stddev = 0.01 )),  
                    'w9'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42                               ], stddev = 0.01 )),  
                    'w10'  : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42                               ], stddev = 0.01 )),  
                    'w11'  : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42                               ], stddev = 0.01 )),   
                    'w12'  : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42                               ], stddev = 0.01 )),  
                    'w13'  : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42                               ], stddev = 0.01 )),   
                    'w14'  : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42                               ], stddev = 0.01 )),   
                    'w15'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 42 , 42                               ], stddev = 0.01 )),  
                    'w16'  : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42                               ], stddev = 0.01 )),  
                    'w17'  : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42                               ], stddev = 0.01 )),   
                    'w18'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 42 , 42                               ], stddev = 0.01 )),  
                    'w19'  : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42                               ], stddev = 0.01 )),   
                    'wOut' : tf.Variable(tf.truncated_normal( [ 1 , 1 , 42 , classes_count*halfed_scene_shape ], stddev = 0.01 ))
                   } 
        params_cnn_b = {
                    'b1'   : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )),  
                    'b2'   : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )),  
                    'b3'   : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b4'   : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b5'   : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b6'   : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b7'   : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b8'   : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b9'   : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b10'  : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b11'  : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b12'  : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b13'  : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b14'  : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b15'  : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b16'  : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b17'  : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b18'  : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'b19'  : tf.Variable(tf.truncated_normal( [ 42                               ], stddev = 0.01 )), 
                    'bOut' : tf.Variable(tf.truncated_normal( [ classes_count*halfed_scene_shape ], stddev = 0.01 ))
                   } 
        params_pix_w = {
                    'w1'   : tf.Variable(tf.truncated_normal( [ 5 , 5 , 1  , 16 ], stddev = 0.01 )),  
                    'w2'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 16 , 32 ], stddev = 0.01 )),  
                    'w3'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 32 , 16 ], stddev = 0.01 )),
                    'w4'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 16 , 16 ], stddev = 0.01 )),  
                    'w5'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 16 , 32 ], stddev = 0.01 )),  
                    'w6'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 32 , 32 ], stddev = 0.01 )),  
                    'w7'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 32 , 16 ], stddev = 0.01 )),  
                    'w8'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 16 , 32 ], stddev = 0.01 )),  
                    'w9'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 32 , 32 ], stddev = 0.01 )),  
                    'w10'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 32 , 16 ], stddev = 0.01 )),  
                    'w11'  : tf.Variable(tf.truncated_normal( [ 3 , 3 , 16 , 32 ], stddev = 0.01 )),   
                    'w12'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 32 , 16 ], stddev = 0.01 )),  
                    'w13'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 16 , classes_count ], stddev = 0.01 ))
                   } 
        params_pix_b = {
                    'b1'   : tf.Variable(tf.truncated_normal( [ 16 ], stddev = 0.01 )),   
                    'b2'   : tf.Variable(tf.truncated_normal( [ 32 ], stddev = 0.01 )),  
                    'b3'   : tf.Variable(tf.truncated_normal( [ 16 ], stddev = 0.01 )), 
                    'b4'   : tf.Variable(tf.truncated_normal( [ 16 ], stddev = 0.01 )),  
                    'b5'   : tf.Variable(tf.truncated_normal( [ 32 ], stddev = 0.01 )), 
                    'b6'   : tf.Variable(tf.truncated_normal( [ 32 ], stddev = 0.01 )), 
                    'b7'   : tf.Variable(tf.truncated_normal( [ 16 ], stddev = 0.01 )),  
                    'b8'   : tf.Variable(tf.truncated_normal( [ 32 ], stddev = 0.01 )), 
                    'b9'   : tf.Variable(tf.truncated_normal( [ 32 ], stddev = 0.01 )), 
                    'b10'  : tf.Variable(tf.truncated_normal( [ 16 ], stddev = 0.01 )),  
                    'b11'  : tf.Variable(tf.truncated_normal( [ 32 ], stddev = 0.01 )), 
                    'b12'  : tf.Variable(tf.truncated_normal( [ 16 ], stddev = 0.01 )), 
                    'b13'  : tf.Variable(tf.truncated_normal( [ classes_count ], stddev = 0.01 ))
                   }    
        params_2d_3d_w = {
                    'w1'   : tf.Variable(tf.truncated_normal( [ 7 , 7 , 1  , 42 ], stddev = 0.01 )),  
                    'w2'   : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42 ], stddev = 0.01 )),  
                    'w3'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42 ], stddev = 0.01 )),
                    'w4'   : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42 ], stddev = 0.01 )),  
                    'w5'   : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42 ], stddev = 0.01 )),  
                    'w6'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42 ], stddev = 0.01 )),  
                    'w7'   : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42 ], stddev = 0.01 )),  
                    'w8'   : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42 ], stddev = 0.01 )),  
                    'w9'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42 ], stddev = 0.01 )),  
                    'w10'  : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42 ], stddev = 0.01 )),  
                    'w11'  : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , 42 ], stddev = 0.01 )),   
                    'w12'  : tf.Variable(tf.truncated_normal( [ 3 , 3 , 42 , 42 ], stddev = 0.01 )),  
                    'w13'  : tf.Variable(tf.truncated_normal( [ 5 , 5 , 42 , classes_count*scene_shape[1] ], stddev = 0.01 ))
                   } 
        params_2d_3d_b = {
                    'b1'   : tf.Variable(tf.truncated_normal( [ 42 ], stddev = 0.01 )),   
                    'b2'   : tf.Variable(tf.truncated_normal( [ 42 ], stddev = 0.01 )),  
                    'b3'   : tf.Variable(tf.truncated_normal( [ 42 ], stddev = 0.01 )), 
                    'b4'   : tf.Variable(tf.truncated_normal( [ 42 ], stddev = 0.01 )),  
                    'b5'   : tf.Variable(tf.truncated_normal( [ 42 ], stddev = 0.01 )), 
                    'b6'   : tf.Variable(tf.truncated_normal( [ 42 ], stddev = 0.01 )), 
                    'b7'   : tf.Variable(tf.truncated_normal( [ 42 ], stddev = 0.01 )),  
                    'b8'   : tf.Variable(tf.truncated_normal( [ 42 ], stddev = 0.01 )), 
                    'b9'   : tf.Variable(tf.truncated_normal( [ 42 ], stddev = 0.01 )), 
                    'b10'  : tf.Variable(tf.truncated_normal( [ 42 ], stddev = 0.01 )),  
                    'b11'  : tf.Variable(tf.truncated_normal( [ 42 ], stddev = 0.01 )), 
                    'b12'  : tf.Variable(tf.truncated_normal( [ 42 ], stddev = 0.01 )), 
                    'b13'  : tf.Variable(tf.truncated_normal( [ classes_count*scene_shape[1] ], stddev = 0.01 ))
                   }
        return params_cnn_w, params_cnn_b, params_pix_w, params_pix_b, params_2d_3d_w, params_2d_3d_b

    #=================================================================================================================================================

    def scoreFun(self): 
    
        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        def conv2d(x, w, b, name="conv_biased", strides=1):
            with tf.name_scope(name):
                x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
                x = tf.nn.bias_add(x, b) 
                return x  
                
        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        def dilated_conv2d(x, w, b, name="d_conv2d", d_rate=1):
            with tf.name_scope(name): 
                x = tf.nn.convolution(x, w, padding='SAME', strides=[1,1], dilation_rate=[d_rate, d_rate], name=name)
                x = tf.nn.bias_add(x, b) 
                return x  
        
        #--------------------------------------------------------------------------------------------------------------------------------------------- 
    
        def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
            with tf.name_scope(layer_name):
                network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
                return network
                
        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        def first_layer(x, scope):
            with tf.name_scope(scope) :
                x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x) 
                return x
        
        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        def transform_layer(x, stride, scope):
            with tf.name_scope(scope) :
                x = conv_layer(x, filter=64, kernel=[1,1], stride=stride, layer_name=scope+'_conv1') 
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x) 
                
                x = conv_layer(x, filter=64, kernel=[3,3], stride=1, layer_name=scope+'_conv2')
                x = tf.layers.batch_normalization(x)
                x = tf.nn.relu(x)
                return x

        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        def transition_layer(x, out_dim, scope):
            with tf.name_scope(scope):
                x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
                x = tf.layers.batch_normalization(x)  
                return x

        #---------------------------------------------------------------------------------------------------------------------------------------------
       
        def split_layer(input_x, stride, layer_name):
            with tf.name_scope(layer_name) :
                layers_split = list()
                for i in range(cardinality) :
                    splits = transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                    layers_split.append(splits)

                return tf.concat(layers_split, axis=3)
                
        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        def residual_layer(input_x, out_dim, layer_num, res_block=blocks): 
            for i in range(res_block): 
                input_dim = int(np.shape(input_x)[-1])
                
                flag = False
                stride = 1
                x = split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i))
                x = transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))
                
                input_x = tf.nn.relu(x + input_x)

            return input_x 
        
        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        def conv2d_pix(x, w, b, name="conv_biased", strides=1, mask_type='B'):
            with tf.name_scope(name):
            
                if mask_type != None: 
                    kh, kw, Cin, Cout = w.shape  
                    kh, kw, Cin, Cout = int(kh), int(kw), int(Cin), int(Cout)  
                    mask   = np.ones((kw, kh, Cin, Cout), dtype=np.float32) 
                    yc, xc = kh // 2, kw // 2 
                    
                    mask[ yc+1: ,     : , : , : ] = 0.0
                    mask[ yc  : , xc+1: , : , : ] = 0.0 
                    
                    if mask_type == 'A':
                        mask[yc, xc, :, :] = 0.
                        
                    w *= mask 
                
                x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
                x = tf.nn.bias_add(x, b) 
                return x 
                
        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        def maxpool2d(x, k=2):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        
        #---SE_CNN-------------------------------------------------------------------------------------------------------------------------------------
        
        self.x_3d = tf.reshape(x_3d, shape = [-1, scene_shape[0], scene_shape[1], halfed_scene_shape]) 
        
        conv_1    = conv2d( self.x_3d, self.params_cnn_w_['w1'], self.params_cnn_b_['b1'], "conv_1" ) 
        
        # Residual Block #1
        conv_r1_1 = tf.layers.batch_normalization(tf.nn.relu( conv_1 )) 
        conv_r1_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r1_1, self.params_cnn_w_['w2'], self.params_cnn_b_['b2'], "conv_r1_2" ) ))   
        conv_r1_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r1_2, self.params_cnn_w_['w3'], self.params_cnn_b_['b3'], "conv_r1_3" ) )) 
        conv_r1_4 =                                           conv2d( conv_r1_3, self.params_cnn_w_['w4'], self.params_cnn_b_['b4'], "conv_r1_4" )  
        merge_1   = tf.add_n([conv_1, conv_r1_4]) 
        
        # Residual Block #2
        conv_r2_1 = tf.layers.batch_normalization(tf.nn.relu( merge_1 ))  
        conv_r2_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r2_1, self.params_cnn_w_['w5'], self.params_cnn_b_['b5'], "conv_r2_2" ) ))   
        conv_r2_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r2_2, self.params_cnn_w_['w6'], self.params_cnn_b_['b6'], "conv_r2_3" ) )) 
        conv_r2_4 =                                           conv2d( conv_r2_3, self.params_cnn_w_['w7'], self.params_cnn_b_['b7'], "conv_r2_4" )  
        merge_2   = tf.add_n([merge_1, conv_r2_4])  
        
        # Residual Block #3
        conv_r3_1 = tf.layers.batch_normalization(tf.nn.relu( merge_2 ))  
        conv_r3_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r3_1, self.params_cnn_w_['w8'],  self.params_cnn_b_['b8'],  "conv_r3_2" ) ))   
        conv_r3_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r3_2, self.params_cnn_w_['w9'],  self.params_cnn_b_['b9'],  "conv_r3_3" ) )) 
        conv_r3_4 =                                           conv2d( conv_r3_3, self.params_cnn_w_['w10'], self.params_cnn_b_['b10'], "conv_r3_4" )   
        merge_3   = tf.add_n([merge_2, conv_r3_4])  
        
        # Residual Block #4
        conv_r4_1 = tf.layers.batch_normalization(tf.nn.relu( merge_3 ))  
        conv_r4_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r4_1, self.params_cnn_w_['w11'], self.params_cnn_b_['b11'], "conv_r4_2" ) ))   
        conv_r4_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r4_2, self.params_cnn_w_['w12'], self.params_cnn_b_['b12'], "conv_r4_3" ) )) 
        conv_r4_4 =                                           conv2d( conv_r4_3, self.params_cnn_w_['w13'], self.params_cnn_b_['b13'], "conv_r4_4" )   
        merge_4   = tf.add_n([merge_3, conv_r4_4]) 
        
        # Residual Block #5
        conv_r5_1 = tf.layers.batch_normalization(tf.nn.relu( merge_4 ))  
        conv_r5_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r5_1, self.params_cnn_w_['w14'], self.params_cnn_b_['b14'], "conv_r5_2" ) ))   
        conv_r5_3 = tf.layers.batch_normalization(tf.nn.relu( dilated_conv2d( conv_r5_2, self.params_cnn_w_['w15'], self.params_cnn_b_['b15'], "conv_r5_3", 2 ) )) 
        conv_r5_4 =                                           conv2d( conv_r5_3, self.params_cnn_w_['w16'], self.params_cnn_b_['b16'], "conv_r5_4" )   
        merge_5   = tf.add_n([merge_4, conv_r5_4]) 
        
        # Residual Block #6
        conv_r6_1 = tf.layers.batch_normalization(tf.nn.relu( merge_5 ))  
        conv_r6_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r6_1, self.params_cnn_w_['w17'], self.params_cnn_b_['b17'], "conv_r6_2" ) ))   
        conv_r6_3 = tf.layers.batch_normalization(tf.nn.relu( dilated_conv2d( conv_r6_2, self.params_cnn_w_['w18'], self.params_cnn_b_['b18'], "conv_r6_3", 4 ) )) 
        conv_r6_4 =                                           conv2d( conv_r6_3, self.params_cnn_w_['w19'], self.params_cnn_b_['b19'], "conv_r6_4" )   
        merge_6   = tf.add_n([merge_5, conv_r6_4]) 
        
        se_cnn_out = conv2d(merge_6, self.params_cnn_w_['wOut'], self.params_cnn_b_['bOut'], "conv_out")   
        
        #---PixelCNN-----------------------------------------------------------------------------------------------------------------------------------
        
        self.x_2d = tf.reshape(x_2d, shape = [-1, scene_shape[0], halfed_scene_shape, 1]) 
        
        conv_1    = conv2d_pix( self.x_2d, self.params_pix_w_['w1'], self.params_pix_b_['b1'], "conv_1" , mask_type='A' )
        
        # Residual Block #1
        conv_r1_1 = tf.nn.relu( conv_1 )  
        conv_r1_2 = tf.nn.relu( conv2d_pix( conv_r1_1, self.params_pix_w_['w2'], self.params_pix_b_['b2'], "conv_r1_2" ) )   
        conv_r1_3 = tf.nn.relu( conv2d_pix( conv_r1_2, self.params_pix_w_['w3'], self.params_pix_b_['b3'], "conv_r1_3" ) ) 
        conv_r1_4 =             conv2d_pix( conv_r1_3, self.params_pix_w_['w4'], self.params_pix_b_['b4'], "conv_r1_4" ) 

        merge_1   = tf.add_n([conv_1, conv_r1_4])  
        
        # Residual Block #2
        conv_r2_1 = tf.nn.relu( merge_1 )  
        conv_r2_2 = tf.nn.relu( conv2d_pix( conv_r2_1, self.params_pix_w_['w5'], self.params_pix_b_['b5'], "conv_r2_2" ) )   
        conv_r2_3 = tf.nn.relu( conv2d_pix( conv_r2_2, self.params_pix_w_['w6'], self.params_pix_b_['b6'], "conv_r2_3" ) ) 
        conv_r2_4 =             conv2d_pix( conv_r2_3, self.params_pix_w_['w7'], self.params_pix_b_['b7'], "conv_r2_4" ) 
        
        merge_2   = tf.add_n([merge_1, conv_r2_4])  
        
        # Residual Block #3
        conv_r3_1 = tf.nn.relu( merge_2 )  
        conv_r3_2 = tf.nn.relu( conv2d_pix( conv_r3_1, self.params_pix_w_['w8'],  self.params_pix_b_['b8'],  "conv_r3_2" ) )   
        conv_r3_3 = tf.nn.relu( conv2d_pix( conv_r3_2, self.params_pix_w_['w9'],  self.params_pix_b_['b9'],  "conv_r3_3" ) ) 
        conv_r3_4 =             conv2d_pix( conv_r3_3, self.params_pix_w_['w10'], self.params_pix_b_['b10'], "conv_r3_4" )  
        
        merge_3   = tf.nn.relu( tf.add_n([merge_2, conv_r3_4]) ) 
        
        conv_2    = tf.nn.relu( conv2d_pix( merge_3, self.params_pix_w_['w11'], self.params_pix_b_['b11'], "conv_2" ) )  
        conv_3    = tf.nn.relu( conv2d_pix( conv_2,  self.params_pix_w_['w12'], self.params_pix_b_['b12'], "conv_3" ) )
        
        merge_4   = tf.nn.relu( tf.add_n([merge_3, conv_3]) ) 
        pix_out   =             conv2d_pix( merge_4,  self.params_pix_w_['w13'], self.params_pix_b_['b13'], "pix_out" ) 
        
        #---2d_3d----------------------------------------------------------------------------------------------------------------------------------- 
        
        # pix_out_r  = tf.reshape(pix_out, [-1, 84, 42, 1, 14])
        # pix_out_r  = tf.argmax (pix_out_r, 4) 
        # pix_out_r  = tf.cast   (pix_out_r, tf.float32) 
        # pix_out_r  = tf.reshape(pix_out_r, [-1, 84, 42, 1])
        
        conv_1   = conv2d( self.x_2d, self.params_2d_3d_w_['w1'], self.params_2d_3d_b_['b1'], "conv_1" ) 
        
        # Residual Block #1
        conv_r1_1 = tf.layers.batch_normalization(tf.nn.relu( conv_1 )) 
        conv_r1_2 = tf.layers.batch_normalization(tf.nn.relu(         conv2d( conv_r1_1, self.params_2d_3d_w_['w2'], self.params_2d_3d_b_['b2'], "conv_r1_2"   ) ))   
        conv_r1_3 = tf.layers.batch_normalization(tf.nn.relu( dilated_conv2d( conv_r1_2, self.params_2d_3d_w_['w3'], self.params_2d_3d_b_['b3'], "conv_r1_3", 2) ))  
        conv_r1_4 =                                                   conv2d( conv_r1_3, self.params_2d_3d_w_['w4'], self.params_2d_3d_b_['b4'], "conv_r1_4"   )  
        merge_1   = tf.add_n([conv_1, conv_r1_4]) 
        
        # Residual Block #2
        conv_r2_1 = tf.layers.batch_normalization(tf.nn.relu( merge_1 ))  
        conv_r2_2 = tf.layers.batch_normalization(tf.nn.relu(         conv2d( conv_r2_1, self.params_2d_3d_w_['w5'], self.params_2d_3d_b_['b5'], "conv_r2_2" ) ))   
        conv_r2_3 = tf.layers.batch_normalization(tf.nn.relu( dilated_conv2d( conv_r2_2, self.params_2d_3d_w_['w6'], self.params_2d_3d_b_['b6'], "conv_r2_3", 4 ) )) 
        conv_r2_4 =                                                   conv2d( conv_r2_3, self.params_2d_3d_w_['w7'], self.params_2d_3d_b_['b7'], "conv_r2_4" )  
        merge_2   = tf.add_n([conv_1, merge_1, conv_r2_4])  
        
        # Residual Block #3
        conv_r3_1 = tf.layers.batch_normalization(tf.nn.relu( merge_2 ))  
        conv_r3_2 = tf.layers.batch_normalization(tf.nn.relu(         conv2d( conv_r3_1, self.params_2d_3d_w_['w8'],  self.params_2d_3d_b_['b8'],  "conv_r3_2" ) ))   
        conv_r3_3 = tf.layers.batch_normalization(tf.nn.relu( dilated_conv2d( conv_r3_2, self.params_2d_3d_w_['w9'],  self.params_2d_3d_b_['b9'],  "conv_r3_3", 8 ) )) 
        conv_r3_4 =                                                   conv2d( conv_r3_3, self.params_2d_3d_w_['w10'], self.params_2d_3d_b_['b10'], "conv_r3_4" )   
        merge_3   = tf.add_n([conv_1, merge_1, merge_2, conv_r3_4])  
        
        # Residual Block #4
        conv_r4_1 = tf.layers.batch_normalization(tf.nn.relu( merge_3 ))  
        conv_r4_2 = tf.layers.batch_normalization(tf.nn.relu(         conv2d( conv_r4_1, self.params_2d_3d_w_['w10'], self.params_2d_3d_b_['b10'], "conv_r4_2" ) ))   
        conv_r4_3 = tf.layers.batch_normalization(tf.nn.relu( dilated_conv2d( conv_r4_2, self.params_2d_3d_w_['w11'], self.params_2d_3d_b_['b11'], "conv_r4_3", 16 ) )) 
        conv_r4_4 =                                                   conv2d( conv_r4_3, self.params_2d_3d_w_['w12'], self.params_2d_3d_b_['b12'], "conv_r4_4" )   
        merge_4   = tf.add_n([conv_1, merge_1, merge_2, merge_3, conv_r4_4]) 
        
        # Residual Block #5
        conv_r5_1 = tf.layers.batch_normalization(tf.nn.relu( merge_4 ))  
        out_2d_3d = conv2d(conv_r5_1, self.params_2d_3d_w_['w13'], self.params_2d_3d_b_['b13'], "out_2d_3d")
        out_2d_3d = tf.transpose(out_2d_3d, perm=[0, 1, 3, 2]) 
        
        #---convert score to 3d scene -----------------------------------------------------------------------------------------------------------------
        
        out_2d_3d_r  = tf.reshape(out_2d_3d, [-1, 84, 44, 14, 42] )
        out_2d_3d_r  = tf.argmax (out_2d_3d_r, 3) 
        out_2d_3d_r  = tf.cast   (out_2d_3d_r, tf.float32)  
        out_2d_3d_r  = tf.reshape(out_2d_3d_r, [-1, 84, 44, 42])
        
        se_cnn_out_r  = tf.reshape(se_cnn_out, [-1, 84, 44, 42, 14] )
        se_cnn_out_r  = tf.argmax (se_cnn_out_r, 4) 
        se_cnn_out_r  = tf.cast   (se_cnn_out_r, tf.float32)  
        se_cnn_out_r  = tf.reshape(se_cnn_out_r, [-1, 84, 44, 42])
        
        #---Smoother-----------------------------------------------------------------------------------------------------------------------------------
        
        input_smoother = tf.add_n([se_cnn_out_r, out_2d_3d_r]) 
        
        res_next_1 = residual_layer(input_smoother, out_dim=42, layer_num='1')
        res_next_2 = residual_layer(res_next_1,     out_dim=42, layer_num='2')
        res_next_3 = residual_layer(res_next_2,     out_dim=42, layer_num='3')
        res_next_4 = residual_layer(res_next_3,     out_dim=42, layer_num='4')
        
        smoother_out = conv_layer(res_next_4, filter=classes_count*halfed_scene_shape, kernel=[1,1], stride=1, layer_name="last_conv") 
        
        return smoother_out, out_2d_3d, pix_out, se_cnn_out
        
    #---------------------------------------------------------------------------------------------------------------------------------------------------

    def costFun(self): 
        
        def focal_loss(labels, logits, gamma=2.0, alpha=4.0): 
            epsilon = 1.e-9
            labels = tf.to_int64(labels)
            labels = tf.convert_to_tensor(labels, tf.int64)
            logits = tf.convert_to_tensor(logits, tf.float32)
            num_cls = logits.shape[1]

            model_out = tf.add(logits, epsilon)
            onehot_labels = tf.one_hot(labels, num_cls)
            ce = tf.multiply(onehot_labels, -tf.log(model_out))
            weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
            fl = tf.multiply(alpha, tf.multiply(weight, ce))
            reduced_fl = tf.reduce_max(fl, axis=1) 
            return reduced_fl
        
        #----------------------------------------------------------------
        
        smoother_out, out_2d_3d, pix_out, se_cnn_out = self.score
        
        #---SE_CNN_LOSS--------------------------------------------------------------- 
        logits = tf.reshape(se_cnn_out, [-1, classes_count])
        labels = tf.reshape(self.y_3d,  [-1               ]) 
        
        se_cnn_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)) 
        se_cnn_loss += tf.reduce_mean(focal_loss(labels, tf.nn.softmax(logits)))
        
        for w in self.params_cnn_w_:
            se_cnn_loss += tf.nn.l2_loss(self.params_cnn_w_[w]) * 0.05 
            
        # penalty term
        logits       = tf.reshape(se_cnn_out, [-1, scene_shape[0], scene_shape[1], halfed_scene_shape, classes_count])
        labels       = tf.reshape(self.y_3d,  [-1, scene_shape[0], scene_shape[1], halfed_scene_shape               ])
        split_logits = tf.split(axis=3, num_or_size_splits=halfed_scene_shape, value=logits)
        split_labels = tf.split(axis=3, num_or_size_splits=halfed_scene_shape, value=labels)
        
        for i in range(1,len(split_logits)):
            se_cnn_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=split_logits[i], labels=split_labels[i-1]))
            
        #---PixelCNN_LOSS--------------------------------------------------------------- 
        logits = tf.reshape(pix_out,   [-1, classes_count])
        labels = tf.reshape(self.y_2d, [-1]) 
        
        pixelcnn_loss  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits, labels=labels ))
        pixelcnn_loss += tf.reduce_mean(focal_loss(labels, tf.nn.softmax(logits)))
        
        for w in self.params_pix_w_ :
            pixelcnn_loss += tf.nn.l2_loss(self.params_pix_w_[w]) * 0.05

        #---2d_3d_LOSS--------------------------------------------------------------- 
        logits = tf.reshape(out_2d_3d, [-1, classes_count])
        labels = tf.reshape(self.y_3d, [-1               ]) 
        
        loss_2d_3d = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)) 
        # loss_2d_3d += tf.reduce_mean(focal_loss(labels, tf.nn.softmax(logits)))
        
        for w in self.params_2d_3d_w_:
            loss_2d_3d += tf.nn.l2_loss(self.params_2d_3d_w_[w]) * 0.05 
            
        # penalty term
        logits       = tf.reshape(out_2d_3d, [-1, scene_shape[0], scene_shape[1], halfed_scene_shape, classes_count])
        labels       = tf.reshape(self.y_3d, [-1, scene_shape[0], scene_shape[1], halfed_scene_shape               ])
        split_logits = tf.split(axis=3, num_or_size_splits=halfed_scene_shape, value=logits)
        split_labels = tf.split(axis=3, num_or_size_splits=halfed_scene_shape, value=labels)
        
        for i in range(1,len(split_logits)):
            loss_2d_3d += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=split_logits[i], labels=split_labels[i-1]))
            
        #---Smoother_LOSS---------------------------------------------------------------
        logits = tf.reshape(smoother_out, [-1, classes_count])
        labels = tf.reshape(self.y_3d,  [-1               ]) 
        
        smoother_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)) 
        # smoother_loss += tf.reduce_mean(focal_loss(labels, tf.nn.softmax(logits)))
        
        # penalty term
        logits       = tf.reshape(smoother_out, [-1, scene_shape[0], scene_shape[1], halfed_scene_shape, classes_count])
        labels       = tf.reshape(self.y_3d, [-1, scene_shape[0], scene_shape[1], halfed_scene_shape               ])
        split_logits = tf.split(axis=3, num_or_size_splits=halfed_scene_shape, value=logits)
        split_labels = tf.split(axis=3, num_or_size_splits=halfed_scene_shape, value=labels)
        
        for i in range(1,len(split_logits)):
            smoother_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=split_logits[i], labels=split_labels[i-1]))
        
        return (smoother_loss + loss_2d_3d + se_cnn_loss)
    #------------------------------------------------------------------------------------------------------------------------------------------------    
    
    def updateFun(self):
        return tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost) 
        
   #--------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, x_3d, x_2d, y_3d, y_2d, lr, keepProb, phase):                    
        self.x_3d        = x_3d
        self.x_2d        = x_2d
        self.y_3d        = y_3d
        self.y_2d        = y_2d
        self.lr          = lr 
        self.keepProb    = keepProb
        self.phase       = phase 

        [self.params_cnn_w_,  self.params_cnn_b_,
        self.params_pix_w_,   self.params_pix_b_,
        self.params_2d_3d_w_, self.params_2d_3d_b_] = ConvNet.paramsFun(self)  
        self.score                                  = ConvNet.scoreFun (self)  
        self.cost                                   = ConvNet.costFun  (self)  
        self.update                                 = ConvNet.updateFun(self)  
     
#=================================================================================================================================================== 

def accuFun(sess, trData, trLabel, trData_2d, batch_size):

    score, _, _, _ = sess.run( ConvNet_class.score , feed_dict={x_3d: trData, x_2d: trData_2d, keepProb: 1.0, phase: False})  
    score          = np.reshape( score,   ( batch_size, scene_shape[0], scene_shape[1], halfed_scene_shape, classes_count ) )  
    trLabel        = np.reshape( trLabel, ( batch_size, scene_shape[0], scene_shape[1], halfed_scene_shape ))   
    
    totalAccuOveral   = 0.0
    totalAccuOccupied = 0.0
    
    for idxBatch in range(0, batch_size): 
        positiveOverall  = 0.0
        positiveOccupied = 0.0 
        totalOccupied    = 0.0
        
        for idx2 in range(0, scene_shape[0]):
            for idx3 in range(0, scene_shape[1]):   
                for idx4 in range(0, halfed_scene_shape):   
                    maxIdxPred = np.argmax(score[idxBatch][idx2][idx3][idx4])  
                    
                    if maxIdxPred == trLabel[idxBatch][idx2][idx3][idx4]:
                        positiveOverall+= 1.0
                        if maxIdxPred > 0:
                            positiveOccupied += 1
                            
                    if trLabel[idxBatch][idx2][idx3][idx4] > 0:
                        totalOccupied+= 1    
                    
        totalAccuOveral += (positiveOverall / (scene_shape[0] * scene_shape[1] * halfed_scene_shape * 1.0))    
        if totalOccupied == 0:
            totalOccupied = (scene_shape[0] * scene_shape[1] * halfed_scene_shape * 1.0)
        totalAccuOccupied += (positiveOccupied / totalOccupied) 
        
    totalAccuOveral   =  totalAccuOveral   / (batch_size * 1.0)
    totalAccuOccupied =  totalAccuOccupied / (batch_size * 1.0)
    
    return totalAccuOveral, totalAccuOccupied

#=================================================================================================================================================== 

def show_result(sess):  
    logging.info("Creating ply files...")
    print       ("Creating ply files...")
    
    bs = 0  
    trData, trLabel   = [], [] 
    batch_arr         = []
    batch_arr_2d      = []
    precision = np.zeros(classes_count)
    recall = np.zeros(classes_count)
    accu1_all, accu2_all = 0.0, 0.0
    
    for item in glob.glob(directory + "/*.ply"):
        os.remove(item)
    
    batch_arr_2d      = []
    batch_arr = []  
    name_arr= []
    
    counter = 0
    for item in glob.glob(test_directory + '*.npy'):
        name_arr.append(str(item[12:]))
        loaded_file = np.load(item)
        batch_arr.append(utils.npy_cutter(loaded_file, scene_shape))
        batch_arr_2d.append(np.load('house_2d/' + str(item[12:])) )
        counter += 1
        
    batch_arr = np.reshape( batch_arr, ( -1, scene_shape[0], scene_shape[1], scene_shape[2] ))
    batch_arr_2d = np.reshape( batch_arr_2d, ( -1, scene_shape[0], scene_shape[2] ))
    trData  = batch_arr[ :, 0:scene_shape[0], 0:scene_shape[1], 0:halfed_scene_shape ]               # input 
    trData_2 = batch_arr_2d[ :, 0:scene_shape[0], halfed_scene_shape:scene_shape[2] ] 
    trLabel = batch_arr[ :, 0:scene_shape[0], 0:scene_shape[1], halfed_scene_shape:scene_shape[2] ]  # gt     
    trData  = np.reshape(trData, (-1, scene_shape[0] * scene_shape[1] * halfed_scene_shape))   
    trData_2 = np.reshape( trData_2, ( -1, scene_shape[0] * halfed_scene_shape )) 
    
    score,_,_,_ = sess.run( ConvNet_class.score , feed_dict={x_3d: trData, x_2d:trData_2, keepProb: 1.0, phase: False})  
    score  = np.reshape(score, (counter, scene_shape[0], scene_shape[1], halfed_scene_shape, classes_count))
    score  = np.argmax(score, 4) 
    trData = np.reshape(trData, (-1, scene_shape[0], scene_shape[1], halfed_scene_shape))
    
    for i in range(counter):   
        trData_i = trData[i,:,:,:]
        trData_i  = np.reshape( trData_i, (scene_shape[0], scene_shape[1], halfed_scene_shape))
        
        score_i = score[i,:,:,:]
        score_i = np.reshape( score_i, (scene_shape[0], scene_shape[1], halfed_scene_shape)) 
        
        empty_scene = np.zeros((84,44,42))
        empty_space = np.zeros((scene_shape[0], scene_shape[1], 50)) 
        empty_scene = np.concatenate((trData_i, empty_scene), axis=2)   
        empty_scene = np.concatenate((empty_scene, empty_space), axis=2)  
        gen_scn = np.concatenate((trData_i, score_i), axis=2)  
        gen_scn = np.concatenate((empty_scene, gen_scn), axis=2)   
        empty_space = np.zeros((scene_shape[0], scene_shape[1], 50))
        gen_scn = np.concatenate((gen_scn, empty_space), axis=2) 
        gen_scn = np.concatenate((gen_scn, batch_arr[i,:,:,:]), axis=2)
        
        output = open( directory + "/" + name_arr[i] + ".ply" , 'w') 
        ply       = ""
        numOfVrtc = 0
        for idx1 in range(gen_scn.shape[0]):
            for idx2 in range(gen_scn.shape[1]): 
                for idx3 in range(gen_scn.shape[2]):  
                    if gen_scn[idx1][idx2][idx3] > 0:  
                        ply = ply + str(idx1)+ " " +str(idx2)+ " " +str(idx3) + str(utils.colors[ int(gen_scn[idx1][idx2][idx3]) ]) + "\n" 
                        numOfVrtc += 1
                        
        output.write("ply"                                   + "\n")
        output.write("format ascii 1.0"                      + "\n")
        output.write("comment VCGLIB generated"              + "\n")
        output.write("element vertex " +  str(numOfVrtc)     + "\n")
        output.write("property float x"                      + "\n")
        output.write("property float y"                      + "\n")
        output.write("property float z"                      + "\n")
        output.write("property uchar red"                    + "\n")
        output.write("property uchar green"                  + "\n")
        output.write("property uchar blue"                   + "\n")
        output.write("property uchar alpha"                  + "\n")
        output.write("element face 0"                        + "\n")
        output.write("property list uchar int vertex_indices"+ "\n")
        output.write("end_header"                            + "\n")
        output.write( ply                                          ) 
        output.close() 
        print       (test_data[i][12:] + ".ply" + " is Done!") 
    
    batch_arr = []  
    batch_arr_2d = []
    name_arr= []
    counter = 0
    
    ###################################################################################################
    
    # logging.info("Creating ply files...")
    # print       ("Creating ply files...")
    # 
    # bs = 0  
    # trData, trLabel   = [], [] 
    # batch_arr         = []
    # batch_arr_2d      = []
    # precision = np.zeros(classes_count)
    # recall = np.zeros(classes_count)
    # accu1_all, accu2_all = 0.0, 0.0
    # 
    # for counter in range(num_of_vis_batch):
    #     trData, trLabel   = [], [] 
    #     batch_arr         = []
    #     batch_arr_2d      = []
    #     bs = 0 
    #     
    #     test_data = utils.fetch_random_batch(train_directory, batch_size)
    #     
    #     for test in test_data:   
    #         batch_arr_2d.append(np.load(str(test)[:7] + "d" + str(test[7:]))) 
    #         
    #         loaded_file = np.load(test)
    #         batch_arr.append(utils.npy_cutter(loaded_file, scene_shape))
    #         bs += 1   
    #         
    #     batch_arr    = np.reshape( batch_arr,    ( bs, scene_shape[0], scene_shape[1], scene_shape[2] ))
    #     batch_arr_2d = np.reshape( batch_arr_2d, ( bs, scene_shape[0], scene_shape[2] ))
    #     trData    = batch_arr[ :, 0:scene_shape[0], 0:scene_shape[1], 0:halfed_scene_shape ]               # input 
    #     trLabel   = batch_arr[ :, 0:scene_shape[0], 0:scene_shape[1], halfed_scene_shape:scene_shape[2] ]  # gt   
    #     trData_2d = batch_arr_2d[ :, 0:scene_shape[0], halfed_scene_shape:scene_shape[2] ]  # input 
    #     trData    = np.reshape(trData,    (-1, scene_shape[0] * scene_shape[1] * halfed_scene_shape))   
    #     trData_2d = np.reshape(trData_2d, (-1, scene_shape[0] * halfed_scene_shape))
    # 
    #     score,_,_,_ = sess.run( ConvNet_class.score , feed_dict={x_3d: trData, x_2d:trData_2d, keepProb: 1.0, phase: False})  
    #     score       = np.reshape( score, ( -1, scene_shape[0], scene_shape[1], halfed_scene_shape, classes_count ))  
    #     score       = np.argmax ( score, 4)     
    #     score       = np.reshape( score, ( -1, scene_shape[0], scene_shape[1], halfed_scene_shape )) 
    #     pre, rec    = utils.precision_recall(score, trLabel, batch_size, classes_count)
    #     precision += pre
    #     recall += rec
    #     
    #     accu1, accu2 = accuFun(sess, trData, trLabel, trData_2d, bs)     
    #     accu1_all += accu1
    #     accu2_all += accu2  
    #     logging.info("A1: %g, A2: %g" % (accu1, accu2))
    #     print       ("A1: %g, A2: %g" % (accu1, accu2))
    #    
    # print precision / num_of_vis_batch * 1.0
    # print recall / num_of_vis_batch * 1.0
    # print accu1_all / num_of_vis_batch * 1.0
    # print accu2_all / num_of_vis_batch * 1.0
    # 
    # for item in glob.glob(directory + "/*.ply"):
    #     os.remove(item)
    # 
    # for i in range(batch_size): 
    #     loaded_file = np.load(test_data[i])
    #     scene = utils.npy_cutter(loaded_file, scene_shape)
    #     trData, trLabel = [], []
    #     
    #     scene_2d = batch_arr_2d[i]
    # 
    #     trData   = scene[ 0:scene_shape[0] , 0:scene_shape[1] , 0:halfed_scene_shape ]               # input 
    #     trData_2 = scene_2d[ 0:scene_shape[0], halfed_scene_shape:scene_shape[2] ]                   # input 
    #     trLabel  = scene[ 0:scene_shape[0] , 0:scene_shape[1] , halfed_scene_shape:scene_shape[2] ]  # gt 
    #     
    #     trData      = np.reshape( trData, ( -1, scene_shape[0] * scene_shape[1] * halfed_scene_shape ))  
    #     trData_2    = np.reshape( trData_2, ( -1, scene_shape[0] * halfed_scene_shape ))  
    #     score,_,_,_ = sess.run( ConvNet_class.score , feed_dict={x_3d: trData, x_2d:trData_2, keepProb: 1.0, phase: False})  
    #     score       = np.reshape( score, ( scene_shape[0], scene_shape[1], halfed_scene_shape, classes_count ))  
    #     score       = np.argmax ( score, 3)     
    #     score       = np.reshape( score, ( scene_shape[0], scene_shape[1], halfed_scene_shape ))
    #     score       = score[0:scene_shape[0], 0:scene_shape[1], 0:halfed_scene_shape]            
    #     trData      = np.reshape( trData, (scene_shape[0], scene_shape[1], halfed_scene_shape))
    #     
    #     gen_scn = np.concatenate((trData, score), axis=2) 
    #     
    #     empty_space = np.zeros((10, scene_shape[1], scene_shape[2]))
    #     gen_scn = np.concatenate((gen_scn, empty_space), axis=0)
    #     gen_scn = np.concatenate((gen_scn, scene), axis=0)
    #     
    #     output = open( directory + "/" + test_data[i][10:] + ".ply" , 'w') 
    #     ply       = ""
    #     numOfVrtc = 0
    #     for idx1 in range(gen_scn.shape[0]):
    #         for idx2 in range(gen_scn.shape[1]): 
    #             for idx3 in range(gen_scn.shape[2]):  
    #                 if gen_scn[idx1][idx2][idx3] > 0:  
    #                     ply = ply + str(idx1)+ " " +str(idx2)+ " " +str(idx3) + str(utils.colors[ int(gen_scn[idx1][idx2][idx3]) ]) + "\n" 
    #                     numOfVrtc += 1
    #                     
    #     output.write("ply"                                   + "\n")
    #     output.write("format ascii 1.0"                      + "\n")
    #     output.write("comment VCGLIB generated"              + "\n")
    #     output.write("element vertex " +  str(numOfVrtc)     + "\n")
    #     output.write("property float x"                      + "\n")
    #     output.write("property float y"                      + "\n")
    #     output.write("property float z"                      + "\n")
    #     output.write("property uchar red"                    + "\n")
    #     output.write("property uchar green"                  + "\n")
    #     output.write("property uchar blue"                   + "\n")
    #     output.write("property uchar alpha"                  + "\n")
    #     output.write("element face 0"                        + "\n")
    #     output.write("property list uchar int vertex_indices"+ "\n")
    #     output.write("end_header"                            + "\n")
    #     output.write( ply                                          ) 
    #     output.close()
    #     logging.info(test_data[i] + ".ply" + " is Done!")
    #     print       (test_data[i] + ".ply" + " is Done!") 
    # 
    # logging.info("A1: %g, A2: %g" % (accu1, accu2))    
    # print       ("A1: %g, A2: %g" % (accu1, accu2))   
    
#===================================================================================================================================================

def fetch_x_y(data, limit):
    batch, x, y          = [], [], []  
    batch_2d, x_2d, y_2d = [], [], []  
    random_batch    = [] 
    
    for i in xrange(batch_size): # randomly fetch batch
        rand_index = random.randint(0, limit-1)
        random_batch.append(data[rand_index])
        batch_2d.append(np.load(train_data_2d[rand_index]))

    for npyFile in random_batch: 
        loaded_scene = np.load(npyFile)
        scene = utils.npy_cutter(loaded_scene, scene_shape) 
        batch.append(scene) 
        
    batch = np.reshape( batch, ( -1, scene_shape[0], scene_shape[1], scene_shape[2] ))    
    x = batch[ : , 0:scene_shape[0] , 0:scene_shape[1], 0:halfed_scene_shape ]               # input 
    y = batch[ : , 0:scene_shape[0] , 0:scene_shape[1], halfed_scene_shape:scene_shape[2] ]  # gt   
    x = np.reshape( x, ( -1, scene_shape[0] * scene_shape[1] * halfed_scene_shape ))
    y = np.reshape( y, ( -1, scene_shape[0] * scene_shape[1] * halfed_scene_shape ))
    
    batch_2d = np.reshape( batch_2d, ( -1, scene_shape[0], scene_shape[2] ))    
    x_2d = batch_2d[ : , 0:scene_shape[0] , 0:halfed_scene_shape ]               # input 
    y_2d = batch_2d[ : , 0:scene_shape[0] , halfed_scene_shape:scene_shape[2] ]  # gt   
    x_2d = np.reshape( x_2d, ( -1, scene_shape[0] * halfed_scene_shape ))
    y_2d = np.reshape( y_2d, ( -1, scene_shape[0] * halfed_scene_shape ))

    return x, y, y_2d, y_2d 

#===================================================================================================================================================
  
if __name__ == '__main__':

    x_3d          = tf.placeholder(tf.float32, [ None, scene_shape[0] * scene_shape[1] * halfed_scene_shape ])
    x_2d          = tf.placeholder(tf.float32, [ None, scene_shape[0] * halfed_scene_shape ])
    y_3d          = tf.placeholder(tf.int32,   [ None, scene_shape[0] * scene_shape[1] * halfed_scene_shape ])   
    y_2d          = tf.placeholder(tf.int32,   [ None, scene_shape[0] * halfed_scene_shape ])
    lr            = tf.placeholder(tf.float32                 )   
    keepProb      = tf.placeholder(tf.float32                 )
    phase         = tf.placeholder(tf.bool                    )
    dropOut       = 0.5
    ConvNet_class = ConvNet(x_3d, x_2d, y_3d, y_2d, lr, keepProb, phase) 
    init_var      = tf.global_variables_initializer() 
    saver         = tf.train.Saver() 
    count_params() # print the number of trainable parameters
    
    # log_device_placement: shows the log of which task will work on which device.
    # allow_soft_placement: TF choose automatically the available device
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:  
        sess.run(init_var) 
        
        # restore model weights
        if to_restore:
            if os.path.exists(directory + '/my-model.meta'): 
                new_saver = tf.train.import_meta_graph(directory + '/my-model.meta')
                new_saver.restore(sess, tf.train.latest_checkpoint(directory)) 
                logging.info("\r\n------------ Saved weights restored. ------------")
                print       ("\r\n------------ Saved weights restored. ------------")
                
        # prevent to add extra node to graph during training        
        tf.get_default_graph().finalize()        
        
        # get the updateable operation from graph for batch norm layer
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)     
        
        # -------------- test phase --------------
        if to_train == False:  
            show_result(sess) 
            logging.info(".ply files are created.!")
            print (".ply files are created.!")
            sys.exit(0)
        
        # -------------- train phase --------------
        step         = 0
        train_cost   = []
        valid_cost   = []
        train_accu1  = []
        train_accu2  = []
        valid_accu1  = []
        valid_accu2  = [] 
        accu1tr, accu2tr = 0, 0
        
        while(step < max_iter):    
        
            x_batch, y_batch, x_batch_2d, y_batch_2d = fetch_x_y(train_data, batch_threshold)  
            
            with tf.control_dependencies(extra_update_ops):  
                cost, _ = sess.run([ConvNet_class.cost, ConvNet_class.update], 
                feed_dict={x_3d: x_batch, y_3d: y_batch, x_2d: x_batch_2d, y_2d:y_batch_2d, lr: learning_rate, keepProb: dropOut, phase: True})    
                train_cost.append(cost) 
            
            # -------------- prints --------------
            if step%1 == 0: 
                logging.info("%s , S:%3g , lr:%g , accu1: %4.3g , accu2: %4.3g , Cost: %2.3g "% ( str(datetime.datetime.now().time())[:-7], step, learning_rate, accu1tr, accu2tr, cost ))
                print       ("%s , S:%3g , lr:%g , accu1: %4.3g , accu2: %4.3g , Cost: %2.3g "% ( str(datetime.datetime.now().time())[:-7], step, learning_rate, accu1tr, accu2tr, cost ))
            
            # -------------- accuracy calculator --------------  
            if step % show_accuracy_step == 0 and show_accuracy:   
                accu1tr, accu2tr = accuFun(sess, x_batch, y_batch, x_batch_2d, batch_size)  
                train_accu1.append(accu1tr)
                train_accu2.append(accu2tr)  
                
            # -------------- save mode, write cost and accuracy --------------  
            if step % save_model_step == 0 and save_model: 
                logging.info("Saving the model...") 
                print       ("Saving the model...") 
                saver.save(sess, directory + '/my-model')
                logging.info("creating cost and accuray plot files...") 
                print       ("creating cost and accuray plot files...")
                utils.write_cost_accuray_plot(directory, train_cost, valid_cost, train_accu1, train_accu2, valid_accu1, valid_accu2) 
                
            # -------------- visualize scenes -------------- 
            if step % visualize_scene_step == 0 and visualize_scene:
                show_result(sess)
                
            # --------------------------------------------- 
            step += 1    
            
        logging.info(" --- \r\n --- \r\n  Trainig process is done after " + str(max_iter) + " iterations. \r\n --- \r\n ---")
        print       (" --- \r\n --- \r\n  Trainig process is done after " + str(max_iter) + " iterations. \r\n --- \r\n ---")
        
#======================================================================================================================================================== 