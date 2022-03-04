#FOR MODIFYING IMAGES AND ARRAYS
import os, cv2
import numpy as np

#KERAS IMPORTS
import keras
from keras.applications.vgg16 import VGG16
from keras.callbacks import ProgbarLogger, EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Conv2DTranspose, Conv2D, core
from keras.preprocessing.image import ImageDataGenerator

#UTILITY GLOBAL VARIABLES
input_dim = [320, 480]  
num_class = 6
C=10
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

#DATAGENERATOR FOR MULTIMODAL SEMANTIC SEGMENTATION
def Segment_datagen(file_path, rgb_args, nir_args, label_args, batch_size, input_size):
    # Create MEMORY enough for one batch of input(s) & labels
    data = np.zeros((2,batch_size,input_size[0],input_size[1],3))
    labels = np.zeros((batch_size,input_size[0]*input_size[1],6))
    # Read the file names
    files = open(file_path)
    names = files.readlines()
    files.close()
    # Enter the indefinite loop of generator
    while True:
        for i in range(batch_size):
            index_of_random_sample = np.random.choice(len(names))
            data[0][i] = fix_size(cv2.imread(rgb_args.data_dir+names[index_of_random_sample].strip('\n')+rgb_args.data_ext), input_size)
            data[1] [i]= fix_size(cv2.imread(nir_args.data_dir+names[index_of_random_sample].strip('\n')+nir_args.data_ext), input_size)
            labels[i] = fix_label(fix_size(cv2.imread(label_args.data_dir+names[index_of_random_sample].strip('\n')+label_args.data_ext), input_size),num_class)
        yield [data[0],data[1]],[labels]

#ARGUMENTS FOR DATA_GENERATOR
RGB_args = gen_args ('/home/vinay/Videos/freiburg_forest_annotated/train/rgb/','.jpg')
NIR_args = gen_args ('/home/vinay/Videos/freiburg_forest_annotated/train/nir_color/','.png')
Label_args = gen_args ('/home/vinay/Videos/freiburg_forest_annotated/train/GT_color/','.png')

# the first argument (file_path) is a text file which contains all the image file names (one in each line)
generator = Segment_datagen(
    file_path = '/home/vinay/Videos/freiburg_forest_annotated/train/rgb/train.txt',
    rgb_args = RGB_args,
    nir_args = NIR_args,
    label_args = Label_args,
    batch_size= 8,
    input_size=input_dim)

#================================================MODEL_ARCHITECTURE============================================================

# RGB MODALITY BRANCH OF CNN
inputs_rgb = Input(shape=(input_dim[0],input_dim[1],3))


# NIR MODALITY BRANCH OF CNN
inputs_nir = Input(shape=(input_dim[0],input_dim[1],3))

concat = keras.layers.concatenate([inputs_rgb,inputs_nir],axis=-1)
conv_init = Conv2D(3, (3,3), strides=(1,1), padding='same', activation='relu', data_format='channels_last')(concat)
vgg_model = VGG16(weights='imagenet', include_top= False, input_tensor=Input(shape=(input_dim[0],input_dim[1],3)))
conv_model = vgg_model(conv_init)
conv_model = Conv2D(1024, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model)
conv_model = Conv2D(1024, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model)
deconv_1 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_model)
#============================================================================================================
conv_1 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_1)
dropout = core.Dropout(0.4)(conv_1)
#===============================================================================================================
deconv_2 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(dropout)
conv_2 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_2)
deconv_3 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_2)
conv_3 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_3)
deconv_4 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_3)
conv_4 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_4)
deconv_5 = Conv2DTranspose(num_class,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_4)
deconv_last = Conv2DTranspose(num_class, (1,1), strides=(1, 1), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal') (deconv_5)


#VECTORIZING OUTPUT
out = core.Reshape((input_dim[0]*input_dim[1],num_class))(deconv_last)


# MODAL [INPUTS , OUTPUTS]
model = Model(inputs=[inputs_rgb,inputs_nir], outputs=[out])
print 'compiling'
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# Save the model according to the conditions  
progbar = ProgbarLogger(count_mode='steps')
checkpoint = ModelCheckpoint("nir_rgb_segmentation_1.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=1, mode='auto')


model.fit_generator(generator,steps_per_epoch=2000,epochs=50, callbacks=[progbar,checkpoint,early])
