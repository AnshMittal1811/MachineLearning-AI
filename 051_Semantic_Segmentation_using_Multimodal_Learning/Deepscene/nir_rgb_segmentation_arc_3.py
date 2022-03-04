#FOR MODIFYING IMAGES AND ARRAYS
import os, cv2
import numpy as np

#KERAS IMPORTS
import keras
from keras.applications.vgg16 import VGG16
from keras.callbacks import ProgbarLogger, EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Conv2DTranspose, Conv2D, core
from keras.preprocessing.image import *

#UTILITY GLOBAL VARIABLES
input_dim = [512, 928]  
num_class = 6
C = 10
index = [2380, 1020,  969,  240, 2775,    0]

#HELPER FUNCTION OF SEGMENT_DATA_GENERATOR
# comprises of path and extension of images in a directory
class gen_args:
    data_dir = None
    data_ext = None
    def __init__(self,dirr,ext):
        self.data_dir = dirr
        self.data_ext = ext
        

#RESIZES 3D IMAGES(image)(EX: RGB) TO DESIRED SIZE(crop_size) 
def fix_size(image, crop_size):
    cropy, cropx = crop_size
    height, width = image.shape[:-1]
    
    #adjusting height of the image 
    cy = cropy - height
    if cy > 0:
        if cy % 2 == 0:
            image = np.vstack((np.zeros((cy/2,width,3)) , image , np.zeros((cy/2,width,3))))
        else:
            image = np.vstack((np.zeros((cy/2,width,3)) , image , np.zeros((cy/2 +1,width,3))))
    if cy < 0:
        if cy % 2 == 0:
            image = np.delete(image, range(-1*cy/2), axis = 0)
            image = np.delete(image, range(height + cy,height +  cy/2), axis = 0)
        else:
            image = np.delete(image, range(-1*cy/2), axis =0)
            image = np.delete(image, range(height + cy, height + cy/2 + 1), axis=0)
    
    #adjusting width of the image
    height, width = image.shape[:-1]
    cx = cropx - width
    if cx > 0:
        if cx % 2 == 0:
            image = np.hstack((np.zeros((height,cx/2,3)) , image , np.zeros((height,cx/2,3))))
        else:
            image = np.hstack((np.zeros((height,cx/2,3)) , image , np.zeros((height,cx/2 + 1,3))))
    if cx < 0:
        if cx % 2 == 0:
            image = np.delete(image, range(-1*cx/2), axis = 1)
            image = np.delete(image, range(width + cx,width +  cx/2), axis = 1)
        else:
            image = np.delete(image, range(-1*cx/2), axis =1)
            image = np.delete(image, range(width + cx, width + cx/2 + 1), axis=1)
    return image


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
class aug_state:
    def __init__(self,flip_axis_index=0,rotation_range=360,height_range=0.2,width_range=0.2,shear_intensity=1,color_intensity=40,zoom_range=(1.2,1.2)):
         self.flip_axis_index=flip_axis_index
         self.rotation_range=rotation_range
         self.height_range=height_range
         self.width_range=width_range
         self.shear_intensity=shear_intensity
         self.color_intensity=color_intensity
         self.zoom_range=zoom_range


def data_augmentor(x,state,row_axis=0,col_axis=1,channel_axis=-1,
    bool_flip_axis=True,
    bool_random_rotation=True,
    bool_random_shift=True,
    bool_random_shear=True,
    bool_random_channel_shift=True,
    bool_random_zoom=True):
    if bool_flip_axis:
        flip_axis(x, state.flip_axis_index)

    if bool_random_rotation:
        random_rotation(x, state.rotation_range, row_axis, col_axis, channel_axis)

    if bool_random_shift:
        random_shift(x, state.width_range, state.height_range, row_axis, col_axis, channel_axis)

    if bool_random_shear:
        random_shear(x, state.shear_intensity, row_axis, col_axis, channel_axis)

    if bool_random_zoom:
        random_zoom(x, state.zoom_range, row_axis, col_axis, channel_axis)

    if bool_random_channel_shift:
        random_channel_shift(x, state.color_intensity, channel_axis)

    return x



#=====================================================================================================================
#DATAGENERATOR FOR MULTIMODAL SEMANTIC SEGMENTATION
def Segment_datagen(state_aug,file_path, rgb_args, nir_args, label_args, batch_size, input_size):
    # Create MEMORY enough for one batch of input(s) + augmentation & labels + augmentation
    data = np.zeros((2,batch_size*2,input_size[0],input_size[1],3))
    labels = np.zeros((batch_size*2,input_size[0]*input_size[1],6))
    # Read the file names
    files = open(file_path)
    names = files.readlines()
    files.close()
    # Enter the indefinite loop of generator
    while True:
        for i in range(batch_size*2):
            index_of_random_sample = np.random.choice(len(names))
            np.random.seed(i)
            data[0][i] = fix_size(cv2.imread(rgb_args.data_dir+names[index_of_random_sample].strip('\n')+rgb_args.data_ext), input_size)
            data[0][batch_size*2-1-i] = data_augmentor(data[0][i],state_aug)
            np.random.seed(i)
            data[1][i]= fix_size(cv2.imread(nir_args.data_dir+names[index_of_random_sample].strip('\n')+nir_args.data_ext), input_size)
            data[1][batch_size*2-1-i] = data_augmentor(data[1][i],state_aug)
            np.random.seed(i)
            temp = fix_size(cv2.imread(label_args.data_dir+names[index_of_random_sample].strip('\n')+label_args.data_ext), input_size)
            labels[i] = fix_label(temp,num_class)
            labels[batch_size*2-1-i] = fix_label(data_augmentor(temp, state_aug, bool_random_channel_shift=False),num_class)
        yield [data[0],data[1]],[labels]

#ARGUMENTS FOR DATA_GENERATOR
RGB_args = gen_args ('/home/vinay/Videos/freiburg_forest_annotated/train/rgb/','.jpg')
NIR_args = gen_args ('/home/vinay/Videos/freiburg_forest_annotated/train/nir_color/','.png')
Label_args = gen_args ('/home/vinay/Videos/freiburg_forest_annotated/train/GT_color/','.png')
state_aug = aug_state() 

generator = Segment_datagen(state_aug,
    file_path = '/home/vinay/Videos/freiburg_forest_annotated/train/rgb/train.txt',
    rgb_args = RGB_args,
    nir_args = NIR_args,
    label_args = Label_args,
    batch_size= 8,
    input_size=input_dim)


#---------------------------------lamda layers for handling gating-----------------------------------------



#================================================MODEL_ARCHITECTURE============================================================

# RGB MODALITY BRANCH OF CNN
inputs_rgb = Input(shape=(input_dim[0],input_dim[1],3))
vgg_model_rgb = VGG16(weights='imagenet', include_top= False)
conv_model_rgb = vgg_model_rgb(inputs_rgb)
conv_model_rgb = Conv2D(1024, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_rgb)
conv_model_rgb_a = Conv2D(1024, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_rgb)
deconv_rgb_1 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_model_rgb_a)
#============================================================================================================
conv_rgb_1 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_rgb_1)
dropout_rgb = core.Dropout(0.4)(conv_rgb_1)
#===============================================================================================================
deconv_rgb_2 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(dropout_rgb)
conv_rgb_2 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_rgb_2)
deconv_rgb_3 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_rgb_2)
conv_rgb_3 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_rgb_3)
deconv_rgb_4 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_rgb_3)
conv_rgb_4 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_rgb_4)
deconv_rgb_5 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_rgb_4)


# NIR MODALITY BRANCH OF CNN
inputs_nir = Input(shape=(input_dim[0],input_dim[1],3))
vgg_model_nir = VGG16(weights='imagenet', include_top= False)
conv_model_nir = vgg_model_rgb(inputs_nir)
conv_model_nir = Conv2D(1024, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_nir)
conv_model_nir_a = Conv2D(1024, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_nir)
deconv_nir_1 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_model_nir_a)
#============================================================================================================
conv_nir_1 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_nir_1)
dropout_nir = core.Dropout(0.4)(conv_nir_1)
#===============================================================================================================
deconv_nir_2 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(dropout_nir)
conv_nir_2 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_nir_2)
deconv_nir_3 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_nir_2)
conv_nir_3 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_nir_3)
deconv_nir_4 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_nir_3)
conv_nir_4 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_nir_4)
deconv_nir_5 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_nir_4)


#------------------------------------ the adaptive gating network------------------------------------------------

# CONACTENATE the features of RGB & NIR 
adaptive_merge = keras.layers.concatenate([conv_model_rgb_a, conv_model_nir_a], axis=-1)

adaptive_conv = Conv2D(num_class, (3,3), strides=(2,2), padding = 'same', activation='relu', data_format='channels_last')(adaptive_merge)
adaptive_conv = Conv2D(1, (3,3), strides=(2,2), padding = 'same', activation='relu', data_format='channels_last')(adaptive_conv)
adaptive_vec = core.Reshape((1,-1))(adaptive_conv)
soft_dense = Dense(2,activation = 'softmax')(adaptive_vec)

#------------------------------------------------------------------------------------------
inshape = deconv_nir_5._keras_shape
print inshape
before_merge_rgb = core.Flatten()(deconv_rgb_5)
print before_merge_rgb._keras_shape
before_merge_nir = core.Flatten()(deconv_nir_5)
print  before_merge_nir._keras_shape
merge_flat = keras.layers.concatenate([before_merge_rgb,before_merge_nir])
print merge_flat._keras_shape

soft_flat = core.Flatten()(soft_dense)
print soft_flat._keras_shape
repeat = core.RepeatVector(before_merge_nir._keras_shape[1])(soft_flat)
print repeat._keras_shape
repeat_flat = core.Flatten()(repeat)
print  repeat_flat._keras_shape

reshape_now = keras.layers.multiply([repeat_flat, merge_flat])
reshape_now = core.Reshape((2,-1))(reshape_now)
outshape =  reshape_now._keras_shape

layer1 = core.Lambda(lambda x: x[:,0:1,:], output_shape=lambda x: (outshape[0],1, outshape[2]))(reshape_now)
layer2 = core.Lambda(lambda x: x[:,1:2,:], output_shape=lambda x: (outshape[0],1, outshape[2]))(reshape_now)


#-------------------------------------------------------------------------------------------------------------------------------
# CONACTENATE the ends of RGB & NIR 
merge_rgb_nir = keras.layers.add([layer1,layer2])
print merge_rgb_nir._keras_shape
#merge_rgb_nir = keras.layers.merge([soft_dense,before_merge_rgb,before_merge_nir], mode=scalarmult)
merge_rgb_nir = core.Flatten()(merge_rgb_nir)
merge_rgb_nir = core.Reshape((inshape[1],inshape[2],inshape[3]))(merge_rgb_nir)

# DECONVOLUTION Layers
deconv_last = Conv2DTranspose(num_class, (1,1), strides=(1, 1), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal') (merge_rgb_nir)

#VECTORIZING OUTPUT
out_reshape = core.Reshape((input_dim[0]*input_dim[1],num_class))(deconv_last)
out = core.Activation('softmax')(out_reshape)

# MODAL [INPUTS , OUTPUTS]
model = Model(inputs=[inputs_rgb,inputs_nir], outputs=[out, soft_dense])
print 'compiling'
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

"""
# Save the model according to the conditions  
progbar = ProgbarLogger(count_mode='steps')
checkpoint = ModelCheckpoint("nir_rgb_segmentation_2.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=1, mode='auto')


model.fit_generator(generator,steps_per_epoch=2000,epochs=50, callbacks=[progbar,checkpoint,early])
"""