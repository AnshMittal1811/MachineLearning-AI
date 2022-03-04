
#FOR MODIFYING IMAGES AND ARRAYS
from datetime import datetime
import os,cv2
#from cv2 import getRotationMatrix2D, warpAffine,getAffineTransform,resize,imread,BORDER_REFLECT
import numpy as np
#KERAS IMPORTS
from keras.applications.vgg16 import VGG16
from keras.callbacks import ProgbarLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Conv2DTranspose, Conv2D, concatenate
from keras.layers.core import Reshape, Activation, Dropout
from keras.preprocessing.image import *
from keras.optimizers import SGD


#UTILITY GLOBAL VARIABLES
input_dim = (512,928)
dim_tup = (928,512)
num_class = 6
C = 4
index = [0, 1020, 1377  , 240, 735, 2380]
#HELPER FUNCTION OF SEGMENT_DATA_GENERATOR
# comprises of path and extension of images in a directory
class gen_args:
    data_dir = None
    data_ext = None
    def __init__(self,dirr,ext):
        self.data_dir = dirr
        self.data_ext = ext
        
#CONVERTING MASKED IMAGES(image) TO A ARRAY OF PIXELWISE ONE-HOT VECTORS(of dimension 'no_class')
def fix_label(image, no_class):
    width , height, depth = image.shape
    #generating hashes for each pixel (index array above has the hash values for each class)
    image = np.dot(image.reshape(width*height,depth)[:,],[1,4,9])
    #converting hashes to indices of classes
    for i in range(no_class):
        image[image == index[i]] = i
    #converting each index into one-hot vector of dim of classes(no_class)
    image = (np.arange(no_class) == image[...,None])*1
    return image


#====================================================data==augmentation==============================================================
'''class aug_state:
    def __init__(self,flip_axis_index=0,zoom_range=(1.2,1.2)):
         self.flip_axis_index=flip_axis_index
         self.zoom_range=zoom_range
         
def data_augmentor(x,state,row_axis=1,col_axis=0,channel_axis=-1):
    #dt = datetime.now()
    #(int(str(dt).split('.')[1])%100)
    t = np.random.randint(4,size=2)
    temp =[0,0,0,0,0]
    temp[t[0]] = 1
    temp[t[1]] = 1
    #print temp
    if temp[0]:
        x = flip_axis(x, state.flip_axis_index)
  
    if temp[1]:
        M = cv2.getRotationMatrix2D((x.shape[1]/2,x.shape[0]/2),np.random.randint(360),1)   #last argument is scale in rotation
        x = cv2.warpAffine(x,M,(x.shape[1],x.shape[0]), borderMode=cv2.BORDER_REFLECT)
	#del M        

    if temp[2]:
        M = np.float32([[1,0,np.random.randint(x.shape[0])],[0,1,np.random.randint(x.shape[1])]])
        x = cv2.warpAffine(x,M,(x.shape[1],x.shape[0]), borderMode = cv2.BORDER_REFLECT)
        #del M

    if temp[3]:
        pts1 = np.float32([[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])],[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])],[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])]])
        pts2 = np.float32([[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])],[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])],[np.random.randint(x.shape[0]),np.random.randint(x.shape[1])]])
        M = cv2.getAffineTransform(pts1,pts2)
        x = cv2.warpAffine(x,M,(x.shape[1],x.shape[0]),borderMode = cv2.BORDER_REFLECT)
        #del M
	#del pts1
	#:del pts2

    if 0:
        x = random_zoom(x, state.zoom_range, row_axis, col_axis, channel_axis,fill_mode='reflect')
        x = np.swapaxes(x,0,1)
        x = np.swapaxes(x,1,2)
        

    return x
'''

#=====================================================================================================================
#DATAGENERATOR FOR MULTIMODAL SEMANTIC SEGMENTATION
def Segment_datagen(file_path, rgb_args, label_args, batch_size, input_size,val_flag):
    # Create MEMORY enough for one batch of input(s) + augmentation & labels + augmentation
    data = np.zeros((batch_size,input_size[0],input_size[1],3), dtype=np.uint8)
    labels = np.zeros((batch_size,input_size[0]*input_size[1],6), dtype=np.uint8)
    # Read the file names
    files = open(file_path)
    names = files.readlines()
    files.close()
    # Enter the indefinite loop of generator
    while True:
	dt = datetime.now()
        np.random.seed(int(str(dt).split('.')[1])%100)
        rand_inds = np.random.random_integers(0,len(names)-1, size=batch_size)
	for i in range(batch_size):
	  	flag = np.random.randint(4)

		if flag or val_flag:
			data[i] = cv2.resize(cv2.imread(rgb_args.data_dir+names[rand_inds[i]].strip('\n')+rgb_args.data_ext), dim_tup)
			labels[i] = fix_label(cv2.resize(cv2.imread(label_args.data_dir+names[rand_inds[i]].strip('\n')+label_args.data_ext), dim_tup),num_class)

		else:			
			num = bin(np.random.randint(1,64))[2:]
			num = '0'*(6-len(num))+num

			data[i] = cv2.resize(cv2.imread(rgb_args.data_dir+'Augmented/'+names[rand_inds[i]].strip('\n')+'_'+num+rgb_args.data_ext), dim_tup)
			labels[i] = fix_label(cv2.resize(cv2.imread(label_args.data_dir+'Augmented/'+names[rand_inds[i]].strip('\n')+'_'+num+label_args.data_ext), dim_tup),num_class)

	
	yield [data],[labels]


#ARGUMENTS FOR DATA_GENERATOR
#state_aug = aug_state() 

#================================================Generator Instances============================================================
train_RGB_args = gen_args ('/home/krishna/freiburg_forest_dataset/train/rgb/','.jpg')
train_Label_args = gen_args ('/home/krishna/freiburg_forest_dataset/train/GT_color/','.png')

train_generator = Segment_datagen(
    file_path = '/home/krishna/freiburg_forest_dataset/train/train.txt',
    rgb_args = train_RGB_args,
    label_args = train_Label_args,
    batch_size= 1,
    input_size=input_dim,
    val_flag = False)



valid_RGB_args = gen_args ('/home/krishna/freiburg_forest_dataset/valid/rgb/','.jpg')
valid_Label_args = gen_args ('/home/krishna/freiburg_forest_dataset/valid/GT_color/','.png')

valid_generator = Segment_datagen(
    file_path = '/home/krishna/freiburg_forest_dataset/valid/valid.txt',
    rgb_args = valid_RGB_args,
    label_args = valid_Label_args,
    batch_size= 1,
    input_size=input_dim,
    val_flag = True)


#================================================MODEL_ARCHITECTURE============================================================

# RGB MODALITY BRANCH OF CNN
inputs_rgb = Input(shape=(input_dim[0],input_dim[1],3))
vgg_model_rgb = VGG16(weights='imagenet', include_top = False,modality_num=0)
conv_model_rgb = vgg_model_rgb(inputs_rgb)
conv_model_rgb = Conv2D(32, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_rgb)
conv_model_rgb = Conv2D(64, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_rgb)
conv_model_rgb = Conv2D(128, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_rgb)
conv_model_rgb = Conv2D(256, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_rgb)
dropout_rgb = Dropout(0.2)(conv_model_rgb)

# DECONVOLUTION Layers
deconv_last = Conv2DTranspose(num_class, (64,64), strides=(32, 32), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal') (dropout_rgb)

#VECTORIZING OUTPUT
out_reshape = Reshape((input_dim[0]*input_dim[1],num_class))(deconv_last)
out = Activation('softmax')(out_reshape)

# MODAL [INPUTS , OUTPUTS]
model = Model(inputs=[inputs_rgb], outputs=[out])
print 'compiling'
model.compile(optimizer=SGD(lr=0.008, decay=1e-6, momentum=0.9, nesterov=True),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.load_weights('late_fusion_unimodal_99.hdf5')
model.summary()
#================================================TRAINING============================================================
# Save the model according to the conditions  
progbar = ProgbarLogger(count_mode='steps')
checkpoint = ModelCheckpoint("late_fusion_unimodal_{epoch:02d}.hdf5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=1, mode='auto')
board = TensorBoard(log_dir='./logs_training', histogram_freq=2, write_graph=True)


model.fit_generator(train_generator,steps_per_epoch=100,epochs=75, callbacks=[progbar,checkpoint,board],validation_data = valid_generator, validation_steps = 2, max_q_size=4, pickle_safe = True)
