
# ====================================================================================================================

import sys, glob, datetime, time, random, os, os.path, shutil, logging
import numpy            as     np
from   random           import randint
from   numpy            import array 
from   collections      import Counter
from   multiprocessing  import Pool 
import tensorflow       as     tf 
import utils # TODO fix it later

# ====================================================================================================================

classes_count         = 14
scene_shape        = [84, 44, 84] 
halfed_scene_shape = scene_shape[2] / 2   
directory             = 'gan_v1'
to_train              = True
to_restore            = False
show_accuracy         = True
show_accuracy_step    = 500
save_model            = True
save_model_step       = 1000
visualize_scene       = True
visualize_scene_step  = 5000
subset_train          = False 
train_directory       = 'house_2/'  
test_directory        = 'test_data/' 
max_iter              = 500000
learning_rate         = 0.00005
batch_size            = 32 

logging.basicConfig(filename=str(directory)+'.log', level=logging.DEBUG) 

if not os.path.exists(directory):
    os.makedirs(directory)
    
train_data = []  
test_data  = []

for item in glob.glob(train_directory + "*.npy"):
    train_data.append(item)
    
for item in glob.glob(test_directory + "*.npy"):
    test_data.append(item)

batch_threshold = 0
if subset_train:
    batch_threshold = batch_size * visualize_scene_step
else:
    batch_threshold = len(train_data)

# ====================================================================================================================

def gaussian_noise(input, sigma = 0.1): 
    noisy = np.random.normal(0.0, sigma, tf.to_int64(input).get_shape())
    return noisy + input
    
# ====================================================================================================================

def writeCostNaccu(g_loss, d_loss, s_loss, ds_loss): 
    output = open(directory + "/costs.py" , 'w') 
    output.write( "import matplotlib.pyplot as plt" + "\r\n" )
    output.write( "g_loss  = []" + "\r\n" )
    output.write( "d_loss  = []" + "\r\n" )
    output.write( "s_loss  = []" + "\r\n" )
    output.write( "ds_loss = []" + "\r\n" )
    output.write( "steps   = []" + "\r\n" ) 
    for i in range(len(g_loss)):
        output.write( "steps.append("+ str(i) +")" + "\r\n" )
    for i in range(len(g_loss)):
        output.write( "g_loss.append("+ str(g_loss[i]) +")" + "\r\n" )
    output.write( "\r\n \r\n \r\n" )
    for i in range(len(d_loss)): 
        output.write( "d_loss.append("+ str(d_loss[i]) +")" + "\r\n" )  
    for i in range(len(s_loss)): 
        output.write( "s_loss.append("+ str(s_loss[i]) +")" + "\r\n" ) 
    for i in range(len(ds_loss)): 
        output.write( "ds_loss.append("+ str(ds_loss[i]) +")" + "\r\n" ) 
    output.write( "plt.plot( steps , g_loss,  color ='b', lw=1 )         " + "\r\n" ) 
    output.write( "plt.plot( steps , d_loss,  color ='r', lw=1 )         " + "\r\n" ) 
    # output.write( "plt.plot( steps , s_loss,  color ='g', lw=1 )         " + "\r\n" ) 
    # output.write( "plt.plot( steps , ds_loss, color ='y', lw=1 )         " + "\r\n" ) 
    output.write( "plt.xlabel('Epoch', fontsize=14)                      " + "\r\n" )
    output.write( "plt.ylabel('Loss',  fontsize=14)                      " + "\r\n" )
    output.write( "plt.suptitle('Blue: G, Red: D, Green: S, Yellow: DS') " + "\r\n" )
    output.write( "plt.show()                                            " + "\r\n" ) 
    print ("costs.py file is created! \r\n")
    
# ====================================================================================================================

def lrelu(input, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * input + f2 * tf.abs(input)
    
# ====================================================================================================================

def conv2d(x, w, b, name="conv2d", strides=1):
    with tf.name_scope(name):
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b) 
        return x  
        
# ====================================================================================================================

def conv2d_stride_2(x, w, b, name="conv2d", strides=2):
    with tf.name_scope(name):
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b) 
        return x 
        
# ====================================================================================================================

def conv2d_transpose(x, w, b, output_shape, name="conv2d_transpose", strides=2):
    with tf.name_scope(name): 
        x = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1,strides,strides,1])
        x = tf.nn.bias_add(x, b) 
        return x  

# ====================================================================================================================

def d_conv2d(x, w, b, name="d_conv2d", d_rate=1):
    with tf.name_scope(name): 
        x = tf.nn.convolution(x, w, padding='SAME', strides=[1,1], dilation_rate=[d_rate, d_rate], name=name)
        x = tf.nn.bias_add(x, b) 
        return x 
        
# ====================================================================================================================

G_W1   = tf.Variable(tf.truncated_normal( [ 7 , 7 , halfed_scene_shape , 64               ], stddev = 0.01 ))   
G_W2   = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
G_W3   = tf.Variable(tf.truncated_normal( [ 5 , 5 , 64 , 64                               ], stddev = 0.01 )) 
G_W4   = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
G_W5   = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
G_W6   = tf.Variable(tf.truncated_normal( [ 5 , 5 , 64 , 64                               ], stddev = 0.01 ))   
G_W7   = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
G_W8   = tf.Variable(tf.truncated_normal( [ 5 , 5 , 64 , 64                               ], stddev = 0.01 ))   
G_W9   = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
G_W10  = tf.Variable(tf.truncated_normal( [ 5 , 5 , 64 , 64                               ], stddev = 0.01 ))   
G_W11  = tf.Variable(tf.truncated_normal( [ 5 , 5 , 64 , 64                               ], stddev = 0.01 ))    
G_W12  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
G_W13  = tf.Variable(tf.truncated_normal( [ 5 , 5 , 64 , 64                               ], stddev = 0.01 ))    
G_W14  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))    
G_W15  = tf.Variable(tf.truncated_normal( [ 1 , 1 , 64 , 64                               ], stddev = 0.01 ))   
G_W16  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
G_W17  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))    
G_W18  = tf.Variable(tf.truncated_normal( [ 1 , 1 , 64 , 64                               ], stddev = 0.01 ))   
G_W19  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
G_WOut = tf.Variable(tf.truncated_normal( [ 1 , 1 , 64 , classes_count*halfed_scene_shape ], stddev = 0.01 )) 
G_b1   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b2   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b3   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b4   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b5   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b6   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b7   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b8   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b9   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b10  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b11  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b12  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b13  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b14  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b15  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b16  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b17  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b18  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_b19  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
G_bOut = tf.Variable(tf.truncated_normal( [ classes_count*halfed_scene_shape ], stddev = 0.01 )) 

g_params   = [G_W1, G_W2, G_W3, G_W4, G_W5, G_W6, G_b1, G_b2, G_b3, G_b4, G_b5, G_b6, G_b7 ,G_b8 ,G_b9 ,G_b10,G_b11,G_b12,G_b13,G_W7 ,G_W8 ,G_W9 ,G_W10,G_W11,G_W12,G_W13,
              G_W14, G_W15, G_W16, G_W17, G_W18, G_W19, G_WOut, G_b14, G_b15, G_b16, G_b17, G_b18, G_b19, G_bOut]   

def generator(f_half_real, keep_prob):   
    inputs    = tf.reshape( f_half_real, [-1, 84, 44, 42] )   
    conv_1    = conv2d( inputs, G_W1, G_b1, "conv_1" ) 
    
    # Residual Block #1
    conv_r1_1 = tf.layers.batch_normalization(tf.nn.relu( conv_1 )) 
    conv_r1_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r1_1,  G_W2,  G_b2, "conv_r1_2" ) ))   
    conv_r1_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r1_2,  G_W3,  G_b3, "conv_r1_3" ) )) 
    conv_r1_4 =                                           conv2d( conv_r1_3,  G_W4,  G_b4, "conv_r1_4" )  
    merge_1   = tf.add_n([conv_1, conv_r1_4]) 
    
    # Residual Block #2
    conv_r2_1 = tf.layers.batch_normalization(tf.nn.relu( merge_1 ))  
    conv_r2_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r2_1,  G_W5,  G_b5, "conv_r2_2" ) ))   
    conv_r2_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r2_2,  G_W6,  G_b6, "conv_r2_3" ) )) 
    conv_r2_4 =                                           conv2d( conv_r2_3,  G_W7,  G_b7, "conv_r2_4" )  
    merge_2   = tf.add_n([merge_1, conv_r2_4])  
    
    # Residual Block #3
    conv_r3_1 = tf.layers.batch_normalization(tf.nn.relu( merge_2 ))  
    conv_r3_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r3_1,  G_W8,   G_b8,  "conv_r3_2" ) ))   
    conv_r3_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r3_2,  G_W9,   G_b9,  "conv_r3_3" ) )) 
    conv_r3_4 =                                           conv2d( conv_r3_3,  G_W10,  G_b10, "conv_r3_4" )   
    merge_3   = tf.add_n([merge_2, conv_r3_4])  
    
    # Residual Block #4
    conv_r4_1 = tf.layers.batch_normalization(tf.nn.relu( merge_3 ))  
    conv_r4_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r4_1,  G_W10,  G_b10, "conv_r4_2" ) ))   
    conv_r4_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r4_2,  G_W11,  G_b11, "conv_r4_3" ) )) 
    conv_r4_4 =                                           conv2d( conv_r4_3,  G_W12,  G_b12, "conv_r4_4" )   
    merge_4   = tf.add_n([merge_3, conv_r4_4]) 
    
    # Residual Block #5
    conv_r5_1 = tf.layers.batch_normalization(tf.nn.relu( merge_4 ))  
    conv_r5_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r5_1,  G_W13,  G_b13, "conv_r5_2" ) ))   
    conv_r5_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r5_2,  G_W14,  G_b14, "conv_r5_3" ) )) 
    conv_r5_4 =                                           conv2d( conv_r5_3,  G_W15,  G_b15, "conv_r5_4" )   
    merge_5   = tf.add_n([merge_4, conv_r5_4]) 
    
    # Residual Block #6
    conv_r6_1 = tf.layers.batch_normalization(tf.nn.relu( merge_5 ))  
    conv_r6_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r6_1,  G_W16,  G_b16, "conv_r6_2" ) ))   
    conv_r6_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r6_2,  G_W17,  G_b17, "conv_r6_3" ) )) 
    conv_r6_4 =                                           conv2d( conv_r6_3,  G_W18,  G_b18, "conv_r6_4" )   
    merge_6   = tf.add_n([merge_5, conv_r6_4]) 
    
    conv_out  = tf.contrib.layers.flatten(conv2d(merge_6,  G_WOut,  G_bOut, "conv_out"))     
    return conv_out
    
# ====================================================================================================================

D_W1   = tf.Variable(tf.truncated_normal( [ 7 , 7 , halfed_scene_shape , 64               ], stddev = 0.01 ))   
D_W2   = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
D_W3   = tf.Variable(tf.truncated_normal( [ 5 , 5 , 64 , 64                               ], stddev = 0.01 )) 
D_W4   = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
D_W5   = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
D_W6   = tf.Variable(tf.truncated_normal( [ 5 , 5 , 64 , 64                               ], stddev = 0.01 ))   
D_W7   = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
D_W8   = tf.Variable(tf.truncated_normal( [ 5 , 5 , 64 , 64                               ], stddev = 0.01 ))   
D_W9   = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
D_W10  = tf.Variable(tf.truncated_normal( [ 5 , 5 , 64 , 64                               ], stddev = 0.01 ))   
D_W11  = tf.Variable(tf.truncated_normal( [ 5 , 5 , 64 , 64                               ], stddev = 0.01 ))    
D_W12  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
D_W13  = tf.Variable(tf.truncated_normal( [ 5 , 5 , 64 , 64                               ], stddev = 0.01 ))    
D_W14  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))    
D_W15  = tf.Variable(tf.truncated_normal( [ 1 , 1 , 64 , 64                               ], stddev = 0.01 ))   
D_W16  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
D_W17  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))    
D_W18  = tf.Variable(tf.truncated_normal( [ 1 , 1 , 64 , 64                               ], stddev = 0.01 ))   
D_W19  = tf.Variable(tf.truncated_normal( [ 3 , 3 , 64 , 64                               ], stddev = 0.01 ))   
D_WOut = tf.Variable(tf.truncated_normal( [ 384, 1 ], stddev = 0.01 )) 
D_b1   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b2   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b3   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b4   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b5   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b6   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b7   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b8   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b9   = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b10  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b11  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b12  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b13  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b14  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b15  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b16  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b17  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b18  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_b19  = tf.Variable(tf.truncated_normal( [ 64                               ], stddev = 0.01 )) 
D_bOut = tf.Variable(tf.truncated_normal( [ 384 ], stddev = 0.01 )) 

d_params   = [D_W1, D_W2, D_W3, D_W4, D_W5, D_W6, D_b1, D_b2, D_b3, D_b4, D_b5, D_b6, D_b7 ,D_b8 ,D_b9 ,D_b10,D_b11,D_b12,D_b13,D_W7 ,D_W8 ,D_W9 ,D_W10,D_W11,D_W12,D_W13,
              D_W14, D_W15, D_W16, D_W17, D_W18, D_W19, D_WOut, D_b14, D_b15, D_b16, D_b17, D_b18, D_b19, D_bOut]

def discriminator(s_half, keep_prob):      
    y   = tf.reshape(s_half, shape = [-1, scene_shape[0], scene_shape[1], halfed_scene_shape]) 
    
    conv_1    = conv2d_stride_2( y, D_W1, D_b1, "conv_1" ) 
    
    # Residual Block #1
    conv_r1_1 = tf.layers.batch_normalization(tf.nn.relu( conv_1 )) 
    conv_r1_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r1_1, D_W2, D_b2, "conv_r1_2" ) ))   
    conv_r1_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r1_2, D_W3, D_b3, "conv_r1_3" ) )) 
    conv_r1_4 =                                           conv2d( conv_r1_3, D_W4, D_b4, "conv_r1_4" )  
    merge_1   = tf.add_n([conv_1, conv_r1_4]) 
    
    # Residual Block #2
    conv_r2_1 = tf.layers.batch_normalization(tf.nn.relu( merge_1 ))  
    conv_r2_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d_stride_2( conv_r2_1, D_W5, D_b5, "conv_r2_2" ) ))   
    conv_r2_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r2_2, D_W6, D_b6, "conv_r2_3" ) )) 
    conv_r2_4 =                                           conv2d( conv_r2_3, D_W7, D_b7, "conv_r2_4" )  
    merge_2   = tf.add_n([conv_r2_2, conv_r2_4])  
    
    # Residual Block #3
    conv_r3_1 = tf.layers.batch_normalization(tf.nn.relu( merge_2 ))  
    conv_r3_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d_stride_2( conv_r3_1, D_W8,  D_b8,  "conv_r3_2" ) ))   
    conv_r3_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r3_2, D_W9,  D_b9,  "conv_r3_3" ) )) 
    conv_r3_4 =                                           conv2d( conv_r3_3, D_W10, D_b10, "conv_r3_4" )   
    merge_3   = tf.add_n([conv_r3_2, conv_r3_4])  
    
    # Residual Block #4
    conv_r4_1 = tf.layers.batch_normalization(tf.nn.relu( merge_3 ))  
    conv_r4_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d_stride_2( conv_r4_1, D_W10, D_b10, "conv_r4_2" ) ))   
    conv_r4_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r4_2, D_W11, D_b11, "conv_r4_3" ) )) 
    conv_r4_4 =                                           conv2d( conv_r4_3, D_W12, D_b12, "conv_r4_4" )   
    merge_4   = tf.add_n([conv_r4_2, conv_r4_4]) 
    
    # Residual Block #5
    conv_r5_1 = tf.layers.batch_normalization(tf.nn.relu( merge_4 ))  
    conv_r5_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d_stride_2( conv_r5_1, D_W13, D_b13, "conv_r5_2" ) ))   
    conv_r5_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r5_2, D_W14, D_b14, "conv_r5_3" ) )) 
    conv_r5_4 =                                           conv2d( conv_r5_3, D_W15, D_b15, "conv_r5_4" )   
    merge_5   = tf.add_n([conv_r5_2, conv_r5_4]) 
    
    # Residual Block #6
    # conv_r6_1 = tf.layers.batch_normalization(tf.nn.relu( merge_5 ))  
    # conv_r6_2 = tf.layers.batch_normalization(tf.nn.relu( conv2d_stride_2( conv_r6_1, D_W16, D_b16, "conv_r6_2" ) ))   
    # conv_r6_3 = tf.layers.batch_normalization(tf.nn.relu( conv2d( conv_r6_2, D_W17, D_b17, "conv_r6_3" ) )) 
    # conv_r6_4 =                                           conv2d( conv_r6_3, D_W18, D_b18, "conv_r6_4" )   
    # merge_6   = tf.add_n([conv_r6_2, conv_r6_4]) 
    
    flat_logits  = tf.reshape( merge_5, [batch_size, -1 ] ) 
    logits  = tf.matmul (flat_logits, D_WOut) + D_bOut 
    return logits
    
# ====================================================================================================================

def show_result(f_half_real, s_half_real, s_half_fake, batch_size, dataType):

    colors  = []  
    colors.append(" 0 0 0 255  ")     # black      for 0  'empty'
    colors.append(" 139 0 0 255")     # dark red   for 1  'ceiling'
    colors.append(" 0 128 0 255")     # green      for 2  'floor'
    colors.append(" 173 216 230 255") # light blue for 3  'wall'
    colors.append(" 0 0 255 255")     # blue       for 4  'window'
    colors.append(" 255 0 0 255")     # red        for 5  'door'
    colors.append(" 218 165 32 255")  # goldenrod  for 6  'chair'
    colors.append(" 210 180 140 255") # tan        for 7  'bed'
    colors.append(" 128 0   128 255") # purple     for 8  'sofa'
    colors.append(" 0  0 139 255")    # dark blue  for 9  'table'
    colors.append(" 255 255 0 255")   # yellow     for 10 'coffee table'
    colors.append(" 128 128 128 255") # gray       for 11 'shelves'
    colors.append(" 0 100 0 255")     # dark green for 12 'cabinets'
    colors.append(" 255 165 0 255")   # orange     for 13 'furniture'  
    
    # real first half
    f_half_real = f_half_real.reshape(( batch_size, 26, 30, 30            ))  
    f_half_real = np.around(         (( f_half_real + 1.0) / 2.0) * 13     )     
    
    # real second half, add 10 empty planes to the end
    s_half_real = s_half_real.reshape(( batch_size, 26, 30, 30            )) 
    s_half_real = np.around(         (( s_half_real + 1.0) / 2.0) * 13     ) 
    temp        = np.zeros           (( batch_size, 26, 30, 10            ))
    s_half_real = np.concatenate     (( s_half_real, temp ), axis=3        )
    
    # generated second half
    s_half_fake = s_half_fake.reshape(( batch_size, 26, 30, 30, 14        )) 
    s_half_fake = np.argmax(s_half_fake, 4)   # convert from 4D to 3D tensor  
    
    # put first and second half together, real and generated
    results1    = np.concatenate(( f_half_real , s_half_real ), axis=3 )  
    results2    = np.concatenate(( f_half_real , s_half_fake ), axis=3 ) 
    results     = np.concatenate(( results1    , results2    ), axis=3 ) 
    
    temp        = np.zeros      (( batch_size  , 26, 30, 10           ))
    f_half_real = np.concatenate(( f_half_real , temp        ), axis=3 ) 
    results     = np.concatenate(( f_half_real , results     ), axis=3 ) 
    
    for i, item in enumerate(results):   
    
        output    = open( data_directory + "/" + dataType[9:] + "_generated_" + str(i) + ".ply" , 'w') 
        ply       = ""
        numOfVrtc = 0
        
        for idx1 in range(26):
            for idx2 in range(30):    
                for idx3 in range(170): 
                    if item[idx1][idx2][idx3] >= 1:  
                        ply = ply + str(idx1)+ " " +str(idx2)+ " " +str(idx3) + str(colors[ int(item[idx1][idx2][idx3]) ]) + "\n" 
                        numOfVrtc += 1 
        output.write("ply"                                    + "\n")
        output.write("format ascii 1.0"                       + "\n")
        output.write("comment VCGLIB generated"               + "\n")
        output.write("element vertex " +  str(numOfVrtc)      + "\n")
        output.write("property float x"                       + "\n")
        output.write("property float y"                       + "\n")
        output.write("property float z"                       + "\n")
        output.write("property uchar red"                     + "\n")
        output.write("property uchar green"                   + "\n")
        output.write("property uchar blue"                    + "\n")
        output.write("property uchar alpha"                   + "\n")
        output.write("element face 0"                         + "\n")
        output.write("property list uchar int vertex_indices" + "\n")
        output.write("end_header"                             + "\n") 
        output.write( ply                                           ) 
        output.close() 
        # print (str(dataType) + "_generated_" + str(i) + ".ply is Done.!") 

# ====================================================================================================================

def accuFun(sess, batch_size, trLabel, generated_scenes): 

    generated_scenes = generated_scenes.reshape(( batch_size, 26, 30, 30, 14 )) 
    generated_scenes = np.argmax( generated_scenes, 4 )   # convert from 4D to 3D tensor 
    trLabel          = np.around( (((trLabel.reshape(( batch_size, 26, 30, 30 ))) + 1.0) / 2.0) * 13 )
    
    accu1 = np.sum(generated_scenes == trLabel) / 23400.0 
    
    accu2 = 0.0
    for b in range(batch_size):
        tp = 0.0
        allTP = 0.0
        for idx1 in range(26):
            for idx2 in range(30):    
                for idx3 in range(30): 
                    if generated_scenes[b][idx1][idx2][idx3] == trLabel[b][idx1][idx2][idx3] and trLabel[b][idx1][idx2][idx3] > 0:
                        tp += 1
                    if trLabel[b][idx1][idx2][idx3] > 0:
                        allTP += 1
                        
        accu2 += (tp / allTP) if allTP != 0 else (tp / 0.000001)
        
    return (accu1 / (batch_size*1.0)), (accu2 / (batch_size*1.0))
    
# ====================================================================================================================

def fetch_x_y(data, limit):
    batch, x, y = [], [], []  
    random_batch = []
    for i in xrange(batch_size): # randomly fetch batch
        random_batch.append(data[random.randint(0, limit-1)])

    for npyFile in random_batch: 
        loaded_scene = np.load(npyFile)
        scene = utils.npy_cutter(loaded_scene, scene_shape) 
        batch.append(scene)
        
    batch = np.reshape( batch, ( -1, scene_shape[0], scene_shape[1], scene_shape[2] ))  

    return batch

# ====================================================================================================================

def train(): 
    
    # -------------- place holders -------------- 
    f_half_real = tf.placeholder(tf.float32, [None, 84, 44, 42], name="f_half_real" ) 
    s_half_real = tf.placeholder(tf.float32, [None, 84, 44, 42], name="f_half_real" ) 
    g_labels    = tf.placeholder(tf.int32,   [None            ], name="g_labels"    ) 
    keep_prob   = tf.placeholder(tf.float32, name="keep_prob")
    batchSize   = tf.placeholder(tf.int32,   name="batchSize")
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # -------------- runs -------------- 
    s_half_gen  = generator    (              f_half_real, keep_prob ) 
    real_logits = discriminator(s_half_real, keep_prob ) 
    
    s_h_s_temp  = tf.reshape( s_half_gen, [-1, 84, 44, 42, 14] )
    s_h_s_temp  = tf.argmax ( s_h_s_temp, 4                    ) 
    s_h_s_temp  = tf.cast   ( s_h_s_temp, tf.float32           )
    s_h_s_temp  = 2 * (s_h_s_temp / tf.constant(13.0)) - 1 
    fake_logits = discriminator(s_h_s_temp, keep_prob )   
    
    # -------------- discriminator loss -------------- 
    D_loss_real  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like ( real_logits )))
    D_loss_fake  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like( fake_logits )))  
    d_loss       = D_loss_real + D_loss_fake 
    
    # -------------- generator loss -------------- 
    g_logits = tf.reshape(s_half_gen, [-1, 14])
    g_labels = tf.reshape(g_labels,   [-1    ])  
    g_loss_t = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=g_logits, labels=g_labels ))
    
    # Penalty term for G
    s_half_fake_ = tf.reshape( s_half_gen, [-1, 84, 44, 42, 14] )
    s_half_real_ = tf.reshape( g_labels,   [-1, 84, 44, 42,    ] )
    split_logits = tf.split( axis=3, num_or_size_splits=42, value=s_half_fake_ )
    split_labels = tf.split( axis=3, num_or_size_splits=42, value=s_half_real_ )
    for i in range(0,len(split_logits)-1):
        g_loss_t += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=split_logits[i], labels=split_labels[i+1] ))
    
    g_loss = g_loss_t + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like ( fake_logits )))  
    
    # -------------- optimization -------------- 
    d_trainer  = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_params )  
    g_trainer  = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_params )   

    # -------------- initialization --------------
    init  = tf.global_variables_initializer() 
    saver = tf.train.Saver() 
    sess  = tf.Session() 
    sess.run(init)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(directory)
        saver.restore(sess, chkpt_fname)
    else:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.mkdir(directory)
    
    # -------------- test phase --------------
    print("The model is running ...")
    if to_train == False:
        chkpt_fname = tf.train.latest_checkpoint(directory)
        saver.restore(sess, chkpt_fname)
        print("\r\n------------ Saved weights restored. ! ------------") 
        print("\r\n---------------------------------------------------") 
        batch_arr = [] 
        bs        = 0
        for npyFile in glob.glob(data_directory + '/*.npy'): 
            batch_arr.append( np.load(npyFile) )
            bs += 1
        batch_arr = np.reshape( batch_arr, ( bs, 60, 26, 30 ))    
        batch_arr = batch_arr.transpose(0,2,3,1)                        # transpose to 26x30x60 
        batch_arr = 2 * (batch_arr.astype(np.float32) / 13.0) - 1     
        generated_scenes = sess.run( s_half_gen, feed_dict={batchSize: bs, f_half_real: batch_arr[:, :, :, 0:30], keep_prob: np.sum(1.0).astype(np.float32)})
        accu1, accu2     = accuFun(sess, bs, batch_arr[:, :, :, 30:60], generated_scenes)
        print("Accuracy: ", accu1, " Completeness:", accu2)
        
        batch_arr = [] 
        bs        = 1 
        print("Creating .ply files...")
        for npyFile in glob.glob(data_directory + '/*.npy'): 
            batch_arr.append( np.load(npyFile) ) 
            batch_arr = np.reshape( batch_arr, ( bs, 60, 26, 30 ))    
            batch_arr = batch_arr.transpose(0,2,3,1)                        # transpose to 26x30x60 
            batch_arr = 2 * (batch_arr.astype(np.float32) / 13.0) - 1     
            generated_scenes = sess.run( s_half_gen, feed_dict={batchSize: bs, f_half_real: batch_arr[:, :, :, 0:30], keep_prob: np.sum(1.0).astype(np.float32)})
            show_result(batch_arr[:, :, :, 0:30], batch_arr[:, :, :, 30:60], generated_scenes, bs, npyFile) 
            batch_arr = [] 
        print (".ply files are created.!") 
        sys.exit(0)
        
    # -------------- training loop --------------
    
    threshold           = 0.0
    step                = 0
    counter             = 0  
    batch_arr           = [] 
    g_l_plot, d_l_plot  = [], []
    s_l_plot, ds_l_plot = [], []
    accu1, accu2        = 0.0, 0.0
    step_threshold      = 0

    while(step < max_iter):     
        batch_arr = fetch_x_y(train_data, batch_threshold)
        
        batch_arr = np.reshape( batch_arr, ( -1, 84, 44, 84 ))     
        g_gt      = np.zeros((84,44,42), dtype=np.float32)              # gt for smoother 
        g_gt      = batch_arr[:, :, :, 42:84]
        g_gt      = np.reshape(g_gt, (batch_size * 84*44*42))
        batch_arr = 2 * (batch_arr.astype(np.float32) / 13.0) - 1       # normalize between [-1,+1]
        
        d_l, g_l = sess.run([d_loss, g_loss], feed_dict={g_labels: g_gt, batchSize: batch_size, f_half_real: batch_arr[:, :, :, 0:42 ], s_half_real: batch_arr[:, :, :, 42:84 ], keep_prob: np.sum(0.5).astype(np.float32)})
        g_l_plot.append(np.mean(g_l)) 
        d_l_plot.append(np.mean(d_l)) 
            
        # -------------- update G --------------  
        if step < step_threshold:
            sess.run(g_trainer,  feed_dict={g_labels: g_gt, batchSize: batch_size, f_half_real: batch_arr[:, :, :, 0:42 ], s_half_real: batch_arr[:, :, :, 42:84], keep_prob: np.sum(0.5).astype(np.float32)})

        # -------------- update All --------------
        if step >= step_threshold:
            sess.run(g_trainer,  feed_dict={g_labels: g_gt, batchSize: batch_size, f_half_real: batch_arr[:, :, :, 0:42 ], s_half_real: batch_arr[:, :, :, 42:84], keep_prob: np.sum(0.5).astype(np.float32)})
            sess.run(d_trainer,  feed_dict={g_labels: g_gt, batchSize: batch_size, f_half_real: batch_arr[:, :, :, 0:42 ], s_half_real: batch_arr[:, :, :, 42:84], keep_prob: np.sum(0.5).astype(np.float32)})
            
        # -------------- show accuracy -------------- 
        # if step%500 == 0:
            # generated_scenes = sess.run(s_half_gen, feed_dict={f_half_real: batch_arr[:, :, :, aug_idx-30:aug_idx ], keep_prob: np.sum(1.0).astype(np.float32)})
            # accu1, accu2     = accuFun(sess, batch_size, batch_arr[:, :, :, 30:60], generated_scenes) 
        
        # -------------- generate results -------------- 
        # if step%1000 == 0:
            # generated_scenes = sess.run(s_half_gen, feed_dict={f_half_real: batch_arr[0:8, :, :, aug_idx-30:aug_idx], keep_prob: np.sum(1.0).astype(np.float32)})
            # show_result(batch_arr[0:8, :, :, 0:30], batch_arr[0:8, :, :, 30:60], generated_scenes, 8, "train") 
            
            # batch_arr = [] 
            # for npyFile in glob.glob('*.npytest'): 
                # batch_arr.append( np.load(npyFile) )
            # batch_arr = np.reshape( batch_arr, ( 8, 60, 26, 30 ))    
            # batch_arr = batch_arr.transpose(0,2,3,1)                        # transpose to 26x30x60
            # batch_arr = 2 * (batch_arr.astype(np.float32) / 13.0) - 1  
            # generated_scenes = sess.run(s_half_gen, feed_dict={f_half_real: batch_arr[:, :, :, 0:30], keep_prob: np.sum(1.0).astype(np.float32)})
            # show_result(batch_arr[:, :, :, 0:30], batch_arr[:, :, :, 30:60], generated_scenes, 8, "test")
            
            # writeCostNaccu(g_l_plot, d_l_plot, s_l_plot, ds_l_plot)  

        # ----------------------------------------------
        if step%1 == 0:
            print("%s, E:%g, Step:%3g, D:%.3f, G:%.3f, A1:%.3f, A2:%.3f"%(str(datetime.datetime.now().time())[:-7], i, step, np.mean(d_l), np.mean(g_l), accu1, accu2)) 

        step     += 1 
        batch_arr = [] 
    # -------------- save model --------------  
    saver.save(sess, os.path.join(directory, "model") ) 

# ====================================================================================================================

if __name__ == '__main__': 
    train() 