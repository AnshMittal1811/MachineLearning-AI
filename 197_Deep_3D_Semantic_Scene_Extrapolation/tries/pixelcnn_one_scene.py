
#====================================================================================================================================================

import sys, glob, datetime, time, random, os, os.path, shutil, logging
import numpy            as     np
from   random           import randint
from   numpy            import array 
from   collections      import Counter
from   multiprocessing  import Pool 
import tensorflow       as     tf 
import utils # TODO fix it later

#=====================================================================================================================================================

classes_count        = 14
scene_shape          = [84, 84] 
halfed_scene_shape   = scene_shape[1] / 2  
directory            = 'pixelcnn_one_scene'
to_train             = True
to_restore           = True
show_accuracy        = True
show_accuracy_step   = 3000
save_model           = True
save_model_step      = 3000
visualize_scene      = True
visualize_scene_step = 3000
subset_train         = False 
train_directory      = 'house_2d/' 
test_directory       = 'house_2d/'
max_iter             = 500000
learning_rate        = 0.05
batch_size           = 1 
num_of_vis_batch     = 1



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
    
#=====================================================================================================================================================

class ConvNet( object ):

    def paramsFun(self): 
        params_w = {
                    'w1'   : tf.Variable(tf.truncated_normal( [ 5 , 5 , 1  , 16 ], stddev = 0.01 )),  
                    'w2'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 16 , 32 ], stddev = 0.01 )),  
                    'w3'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 32 , 16 ], stddev = 0.01 )),
                    'w4'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 16 , 16 ], stddev = 0.01 )),  
                    'w5'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 16 , 32 ], stddev = 0.01 )),  
                    'w6'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 32 , 32 ], stddev = 0.01 )),  
                    'w7'   : tf.Variable(tf.truncated_normal( [ 1 , 1 , 32 , 16 ], stddev = 0.01 )),  
                    'w8'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 16 , 32 ], stddev = 0.01 )),  
                    'w9'   : tf.Variable(tf.truncated_normal( [ 3 , 3 , 32 , 32 ], stddev = 0.01 )),  
                    'w10'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 32 , 16 ], stddev = 0.01 )),  
                    'w11'  : tf.Variable(tf.truncated_normal( [ 3 , 3 , 16 , 32 ], stddev = 0.01 )),   
                    'w12'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 32 , 16 ], stddev = 0.01 )),  
                    'w13'  : tf.Variable(tf.truncated_normal( [ 1 , 1 , 16 , 14 ], stddev = 0.01 ))
                   }
                    
        params_b = {
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
                    'b13'  : tf.Variable(tf.truncated_normal( [ 14 ], stddev = 0.01 ))
                   }
                    
        return params_w,params_b

    #=================================================================================================================================================

    def scoreFun(self): 
    
        def conv2d(x, w, b, name="conv_biased", strides=1, mask_type='B'):
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
                tf.summary.histogram("weights", w)
                tf.summary.histogram("biases",  b) 
                return x 
                
        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        def maxpool2d(x, k=2):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        
        #---------------------------------------------------------------------------------------------------------------------------------------------
        
        self.x    = tf.reshape(x, shape = [-1, scene_shape[0], scene_shape[1], 1]) 
        
        conv_1    = conv2d( self.x, self.params_w_['w1'], self.params_b_['b1'], "conv_1" , mask_type='A' )
        
        # Residual Block #1
        conv_r1_1 = tf.nn.relu( conv_1 )  
        conv_r1_2 = tf.nn.relu( conv2d( conv_r1_1, self.params_w_['w2'], self.params_b_['b2'], "conv_r1_2" ) )   
        conv_r1_3 = tf.nn.relu( conv2d( conv_r1_2, self.params_w_['w3'], self.params_b_['b3'], "conv_r1_3" ) ) 
        conv_r1_4 =             conv2d( conv_r1_3, self.params_w_['w4'], self.params_b_['b4'], "conv_r1_4" ) 

        merge_1   = tf.add_n([conv_1, conv_r1_4])  
        
        # Residual Block #2
        conv_r2_1 = tf.nn.relu( merge_1 )  
        conv_r2_2 = tf.nn.relu( conv2d( conv_r2_1, self.params_w_['w5'], self.params_b_['b5'], "conv_r2_2" ) )   
        conv_r2_3 = tf.nn.relu( conv2d( conv_r2_2, self.params_w_['w6'], self.params_b_['b6'], "conv_r2_3" ) ) 
        conv_r2_4 =             conv2d( conv_r2_3, self.params_w_['w7'], self.params_b_['b7'], "conv_r2_4" ) 
        
        merge_2   = tf.add_n([merge_1, conv_r2_4])  
        
        # Residual Block #3
        conv_r3_1 = tf.nn.relu( merge_2 )  
        conv_r3_2 = tf.nn.relu( conv2d( conv_r3_1, self.params_w_['w8'],  self.params_b_['b8'],  "conv_r3_2" ) )   
        conv_r3_3 = tf.nn.relu( conv2d( conv_r3_2, self.params_w_['w9'],  self.params_b_['b9'],  "conv_r3_3" ) ) 
        conv_r3_4 =             conv2d( conv_r3_3, self.params_w_['w10'], self.params_b_['b10'], "conv_r3_4" )  
        
        merge_3   = tf.nn.relu( tf.add_n([merge_2, conv_r3_4]) ) 
        
        conv_2    = tf.nn.relu( conv2d( merge_3, self.params_w_['w11'], self.params_b_['b11'], "conv_2" ) )  
        conv_3    = tf.nn.relu( conv2d( conv_2,  self.params_w_['w12'], self.params_b_['b12'], "conv_3" ) )
        
        merge_4   = tf.nn.relu( tf.add_n([merge_3, conv_3]) ) 
        conv_4    =             conv2d( merge_4,  self.params_w_['w13'], self.params_b_['b13'], "conv_4" )  
        
        netOut    = tf.contrib.layers.flatten(conv_4)
        
        return netOut
        
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
    
        logits = tf.reshape(self.score, [-1, classes_count])
        labels = tf.reshape(self.y,     [-1 ]) 
        
        total  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits, labels=labels ))
        total += tf.reduce_mean(focal_loss(labels, tf.nn.softmax(logits)))
        
        for w in self.params_w_ :
            total += tf.nn.l2_loss(self.params_w_[w]) * 0.05   
        
        return total
        
    #------------------------------------------------------------------------------------------------------------------------------------------------    
    
    def updateFun(self):
        return tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.cost)

    #------------------------------------------------------------------------------------------------------------------------------------------------
    
    def sumFun(self):
        return tf.summary.merge_all()
        
   #--------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, x, y, lr, keepProb, phase):                    
        self.x         = x
        self.y         = y
        self.lr        = lr 
        self.keepProb  = keepProb
        self.phase     = phase 

        [self.params_w_, self.params_b_] = ConvNet.paramsFun(self) # initialization and packing the parameters
        self.score                       = ConvNet.scoreFun (self) # Computing the score function     
        self.cost                        = ConvNet.costFun  (self) # Computing the cost function 
        self.update                      = ConvNet.updateFun(self) # Computing the update function
        self.sum                         = ConvNet.sumFun   (self) # summary logger 4 TensorBoard

#===================================================================================================================================================
  
def accuFun(sess, trData, trLabel, batch_size): 
    scenes = generate_for_accu(sess, trData, halfed_scene_shape)  
    scenes = np.reshape(scenes, (batch_size, scene_shape[0], scene_shape[1]))   
    trLabel = np.reshape(trLabel, (batch_size, scene_shape[0], scene_shape[1]))   
    
    totalAccuOveral = 0.0 
    
    for idxBatch in range(batch_size): 
        positiveOverall = 0.0  
        
        for idx2 in range(0, scene_shape[0]):
            for idx3 in range(0, scene_shape[1]):      
            
                if scenes[idxBatch][idx2][idx3] == trLabel[idxBatch][idx2][idx3]:
                    positiveOverall+= 1.0
                    
        totalAccuOveral += (positiveOverall / (scene_shape[0] * scene_shape[1] * 1.0))  
        
    totalAccuOveral /= (batch_size * 1.0) 
    
    return totalAccuOveral 
    
#===================================================================================================================================================

def generate_for_accu(sess, scenes, halfed_scene_shape):  
    scenes = np.reshape(scenes, (batch_size, scene_shape[0], scene_shape[1]))  
    scenes[:, :, halfed_scene_shape:] = 0.  
    
    for i in range(scene_shape[0]):
        for j in range(halfed_scene_shape):
            scenes = np.reshape( scenes, ( -1, scene_shape[0] * scene_shape[1]))
            score = sess.run(ConvNet_class.score, feed_dict={x: scenes, keepProb: 1.0, phase: False})  
            score = np.reshape(score, (batch_size, scene_shape[0], scene_shape[1], 1, classes_count))
            score = np.argmax(score, 4)     
            score = np.reshape(score, (batch_size, scene_shape[0], scene_shape[1], 1))
            scenes = np.reshape(scenes, (batch_size, scene_shape[0], scene_shape[1], 1))
            scenes[:, i, j, 0] = score[:, i, j, 0] 
            
    return scenes 
    
#===================================================================================================================================================

def generate(sess, scenes, halfed_scene_shape):
    for item in glob.glob(directory + "/*.ply"):
        os.remove(item) 
        
    scenes = np.reshape(scenes, (batch_size, scene_shape[0], scene_shape[1]))
    
    for i in range(batch_size):
        utils.npy_to_ply(directory + "/full" + str(i), scenes[i]) 
        
    scenes[:, :, halfed_scene_shape:] = 0.  
    
    for i in range(scene_shape[0]):
        for j in range(halfed_scene_shape):
            scenes = np.reshape( scenes, ( -1, scene_shape[0] * scene_shape[1]))
            score = sess.run(ConvNet_class.score, feed_dict={x: scenes, keepProb: 1.0, phase: False})  
            score = np.reshape(score, (batch_size, scene_shape[0], scene_shape[1], 1, classes_count))
            score = np.argmax(score, 4)     
            score = np.reshape(score, (batch_size, scene_shape[0], scene_shape[1], 1))
            scenes = np.reshape(scenes, (batch_size, scene_shape[0], scene_shape[1], 1))
            scenes[:, i, j, 0] = score[:, i, j, 0] 

    
    
    for i in range(batch_size):
        utils.npy_to_ply(directory + "/halfed" + str(i), scenes[i]) 
    
    return scenes 
    
#===================================================================================================================================================
  
if __name__ == '__main__':

    input         = scene_shape[0] * scene_shape[1]
    out           = scene_shape[0] * scene_shape[1]  
    x             = tf.placeholder(tf.float32, [ None, input ])
    y             = tf.placeholder(tf.int32  , [ None, out   ])   
    lr            = tf.placeholder(tf.float32                 )   
    keepProb      = tf.placeholder(tf.float32                 )
    phase         = tf.placeholder(tf.bool                    )
    dropOut       = 0.5
    ConvNet_class = ConvNet(x, y, lr, keepProb, phase)
    init_var      = tf.global_variables_initializer() 
    saver         = tf.train.Saver() 
    
    # log_device_placement: shows the log of which task will work on which device.
    # allow_soft_placement: TF choose automatically the available device
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:  
        sess.run(init_var) 
        
        # restore model weights
        if to_restore:
            if os.path.exists(directory + '/my-model.meta'): 
                new_saver = tf.train.import_meta_graph(directory + '/my-model.meta')
                new_saver.restore(sess, tf.train.latest_checkpoint(directory)) 
                print       ("\r\n------------ Saved weights restored. ------------")
                
        # prevent to add extra node to graph during training        
        tf.get_default_graph().finalize()        
        
        # get the updateable operation from graph for batch norm layer
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)     
        
        # -------------- test phase --------------
        if to_train == False:  
            show_result(sess) 
            print (".ply files are created.!")
            sys.exit(0)
        
        # -------------- train phase --------------
        step         = 1
        train_cost   = []
        valid_cost   = []
        train_accu1  = []
        train_accu2  = []
        valid_accu1  = []
        valid_accu2  = [] 
        accu1tr, accu2tr = 0, 0
        
        while(step < max_iter):    
        
            x_batch = np.load(train_directory + "/0d341dc58dc8db30eba8db965a5ae8b8_0_24.npy") 
            y_batch = np.load(train_directory + "/0d341dc58dc8db30eba8db965a5ae8b8_0_24.npy")
            
            x_batch = np.reshape(x_batch, (-1, scene_shape[0] * scene_shape[1]))
            y_batch = np.reshape(y_batch, (-1, scene_shape[0] * scene_shape[1]))
            
            with tf.control_dependencies(extra_update_ops):  
                cost, _ = sess.run([ConvNet_class.cost, ConvNet_class.update], feed_dict={x: x_batch, y: y_batch, lr: learning_rate, keepProb: dropOut, phase: True})    
                train_cost.append(cost) 
            
            # -------------- prints --------------
            if step%1 == 0:  
                print       ("%s , S:%3g , lr:%g , accu1: %4.3g , Cost: %2.3g "% ( str(datetime.datetime.now().time())[:-7], step, learning_rate, accu1tr, cost ))
            
            # -------------- accuracy calculator --------------  
            if step % show_accuracy_step == 0 and show_accuracy:   
                accu1tr = accuFun(sess, x_batch, y_batch, batch_size)   
                
            # -------------- visualize scenes -------------- 
            if step % visualize_scene_step == 0 and visualize_scene:
                generate(sess, x_batch, halfed_scene_shape) 
                
            # --------------------------------------------- 
            step += 1     