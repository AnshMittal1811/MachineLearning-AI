import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.compat.v1.keras.layers import BatchNormalization



alpha = 0.5
bn_mode = 'default' 
activ = None
kinit = tf.random_normal_initializer(0., 0.02)
if bn_mode== 'selu':
    kinit = tf.keras.initializers.LecunNormal()
    activ = 'selu'

    

def makeDiscriminator():
    

    input_img = tf.keras.layers.Input(shape=[128, 128, 3], name='input_image')
    
    x = Conv2D(64, 4, strides=2, padding='same',kernel_initializer=kinit, use_bias=False,activation=activ)(input_img)
    
    if activ == None:
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=alpha)(x)
    
    x = Conv2D(128, 4, strides=2, padding='same',kernel_initializer=kinit, use_bias=False,activation=activ) (x)
    if activ == None:
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=alpha)(x)
    
    x = Conv2D(256, 4, strides=2,padding='same',kernel_initializer=kinit,use_bias=False,activation=activ)(x) 
    if activ == None:
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=alpha)(x)
    
    x = Conv2D(512, 4, strides=2,padding='same',kernel_initializer=kinit,use_bias=False,activation=activ)(x) 
    if activ == None:
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=alpha)(x)
    
    x = Flatten()(x)
    
    x = Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=input_img, outputs=x)