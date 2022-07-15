import tensorflow as tf
import os
import sys
sys.path.append('..')
import tools as tools
import numpy as np
import matplotlib.pyplot as plt

GPU='0'


def load_real_rgbs(test_mv=5):
    obj_rgbs_folder ='./Data_sample/amazon_real_rgbs/airfilter/'
    rgbs = []
    rgbs_views = sorted(os.listdir(obj_rgbs_folder))
    for v in rgbs_views:
        if not v.endswith('png'): continue
      
        rgbs.append(tools.Data.load_single_X_rgb_r2n2(obj_rgbs_folder + v, train=False))
    
    rgbs = np.asarray(rgbs)
    x_sample = rgbs[0:test_mv, :, :, :].reshape(1, test_mv, 127, 127, 3)
    return x_sample, None

def load_shapenet_rgbs(test_mv=8):
    obj_rgbs_folder = './Data_sample/ShapeNetRendering/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/rendering/'
    obj_gt_vox_path ='./Data_sample/ShapeNetVox32/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/model.binvox'
    rgbs=[]
    rgbs_views = sorted(os.listdir(obj_rgbs_folder))
    for v in rgbs_views:
        if not v.endswith('png'): continue
        rgbs.append(tools.Data.load_single_X_rgb_r2n2(obj_rgbs_folder + v, train=False))
    rgbs = np.asarray(rgbs)
    x_sample = rgbs[0:test_mv, :, :, :].reshape(1, test_mv, 127, 127, 3)
    y_true = tools.Data.load_single_Y_vox(obj_gt_vox_path)
    return x_sample, y_true

def ttest_demo():
#    model_path = './Model_released/'
    model_path='/home/ajith/3d-reconstruction/attsets/Model_released/'
    if not os.path.isfile(model_path + 'model.cptk.data-00000-of-00001'):
        print ('please download our released model first!')
        return

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.visible_device_list = GPU
    

    
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(model_path + 'model.cptk.meta', clear_devices=True)
        saver.restore(sess, model_path + 'model.cptk')
        print ('model restored!')      
        
       # graph = tf.get_default_graph()
       # print(graph.get_operations())
        
        X = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        Y_pred = tf.get_default_graph().get_tensor_by_name("r2n/Reshape_9:0")
        
        plot_data_8 = tf.get_default_graph().get_tensor_by_name("r2n/Reshape_8:0")
        plot_data_7 = tf.get_default_graph().get_tensor_by_name("r2n/Reshape_7:0") #############(1,1024)
        plot_data_6 = tf.get_default_graph().get_tensor_by_name("r2n/Reshape_6:0") #############(1,1024)
        plot_data_5 = tf.get_default_graph().get_tensor_by_name("r2n/Reshape_5:0")
        plot_data_4 = tf.get_default_graph().get_tensor_by_name("r2n/Reshape_4:0")
        plot_data_3 = tf.get_default_graph().get_tensor_by_name("r2n/Reshape_3:0")
        plot_data_2 = tf.get_default_graph().get_tensor_by_name("r2n/Reshape_2:0")
        plot_data_1 = tf.get_default_graph().get_tensor_by_name("r2n/Reshape_1:0")
   
        
#        print("X: ", X.shape)        #Tensor("Placeholder:0", shape=(?, ?, 127, 127, 3), dtype=float32)
#        print(Y_pred)   #Tensor("r2n/Reshape_9:0", shape=(?, 32, 32, 32), dtype=float32)   
   
#        print("x_sample: ", x_sample.shape)  
#        print("x_sample_data: ", type(x_sample[:,:,:,:,1]))  
#        print(y_pred.shape)    ###############################(1, 32, 32, 32) ##############################


#        x_sample, gt_vox = load_shapenet_rgbs() 
        x_sample, gt_vox = load_real_rgbs()
        
   
        plot_buf_1= tf.reshape(plot_data_1, [-1, 32, 32, 1])
        plot_buf_2= tf.reshape(plot_data_2, [-1, 32, 32, 1])
        plot_buf_3= tf.reshape(plot_data_3, [-1, 32, 32, 1])
        plot_buf_4= tf.reshape(plot_data_4, [-1, 32, 32, 1])
        plot_buf_5= tf.reshape(plot_data_5, [-1, 32, 32, 1])
        plot_buf_6= tf.reshape(plot_data_6, [-1, 32, 32, 1])
        plot_buf_7= tf.reshape(plot_data_7, [-1, 32, 32, 1])
        plot_buf_8= tf.reshape(plot_data_8, [-1, 32, 32, 1])
        
      
#        tf.summary.image("RESHAPE_1", plot_buf_1)
#        tf.summary.image("RESHAPE_2", plot_buf_2)
#        tf.summary.image("RESHAPE_3", plot_buf_3)
#        tf.summary.image("RESHAPE_4", plot_buf_4)
#        tf.summary.image("RESHAPE_5", plot_buf_5)
#        tf.summary.image("RESHAPE_6", plot_buf_6)
#        tf.summary.image("RESHAPE_7", plot_buf_7)
#        tf.summary.image("RESHAPE_8", plot_buf_8)
        
        
        summary_8 = tf.summary.image("RESHAPE_8", plot_buf_8)
        summary_7 = tf.summary.image("RESHAPE_7", plot_buf_7)
        summary_6 = tf.summary.image("RESHAPE_6", plot_buf_6)
        summary_5 = tf.summary.image("RESHAPE_5", plot_buf_5)
        summary_4 = tf.summary.image("RESHAPE_4", plot_buf_4)
        summary_3 = tf.summary.image("RESHAPE_3", plot_buf_3)
        summary_2 = tf.summary.image("RESHAPE_2", plot_buf_2)
        summary_1 = tf.summary.image("RESHAPE_1", plot_buf_1)
    
        
#        summary_op = tf.summary.image("RESHAPE_4", plot_buf_4)
#        with tf.Session() as sess:
#        y_pred,1_summary,2_summary = sess.run([Y_pred,summary_op_1,summary_op_2], feed_dict={X: x_sample})
	
        y_pred,summary_pred_1,summary_pred_2,summary_pred_3,summary_pred_4,summary_pred_5,summary_pred_6,summary_pred_7,summary_pred_8  = sess.run([Y_pred,summary_1,summary_2,summary_3,summary_4,summary_5,summary_6,summary_7,summary_8], feed_dict={X: x_sample})
         
#       Write summary  tf.summary.FileWriter
        writer = tf.summary.FileWriter('./logs')
        
   
        writer.add_summary(summary_pred_1)
        writer.add_summary(summary_pred_2)
        writer.add_summary(summary_pred_3)
        writer.add_summary(summary_pred_4)
        writer.add_summary(summary_pred_5)
        writer.add_summary(summary_pred_6)
        writer.add_summary(summary_pred_7)
        writer.add_summary(summary_pred_8)
        
        writer.close()
        
#       sys.exit(). sys.exit() 
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
	 
	print ('enterd')    
	ttest_demo()
#	with tf.Session() as sess:
    # or creating the writer inside the session
#		merge = tf.summary.merge_all()            
#		writer = tf.summary.FileWriter('./graphs/test', sess.graph)
		
    
