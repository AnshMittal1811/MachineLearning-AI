from datetime import datetime
import os,cv2
import numpy as np
#KERAS IMPORTS
from keras.applications.vgg16 import VGG16
from keras.callbacks import ProgbarLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Conv2DTranspose, Conv2D, concatenate
from keras.layers.core import Reshape, Activation, Dropout
from keras.preprocessing.image import *
from keras.optimizers import SGD

class gen_args:
    data_dir = None
    data_ext = None
    def __init__(self,dirr,ext):
        self.data_dir = dirr
        self.data_ext = ext


input_dim = (512,928)
dim_tup = (928,512)
num_class = 6
C = 4
index = [0, 1020, 1377  , 240, 735, 2380]

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

#===========================================TEST DATA GENERATOR==================================================
def Test_datagen(file_path, rgb_args, label_args, batch_size, input_size):
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
			data[i] = cv2.resize(cv2.imread(rgb_args.data_dir+names[rand_inds[i]].strip('\n')+rgb_args.data_ext), dim_tup)
			labels[i] = fix_label(cv2.resize(cv2.imread(label_args.data_dir+names[rand_inds[i]].strip('\n')+label_args.data_ext), dim_tup),num_class)
	yield [data],[labels]

#================================================TEST Data generator instance============================================================
test_RGB_args = gen_args ('/home/krishna/freiburg_forest_dataset/test/rgb/','.jpg')
test_Label_args = gen_args ('/home/krishna/freiburg_forest_dataset/test/GT_color/','.png')

test_datagen = Test_datagen(
	file_path='/home/krishna/freiburg_forest_dataset/test/test.txt',
	rgb_args = test_RGB_args,
	label_args = test_Label_args,
	batch_size = 1,
	input_size = input_dim)
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
model.load_weights('late_fusion_unimodal_95.hdf5')
model.summary()



a = model.evaluate_generator(generator=test_datagen, steps=200)
b = model.metrics_names


print b
print a 
