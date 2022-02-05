import tensorflow as tf
from tensorflow.compat.v1.keras.layers import BatchNormalization       # reverting to v1 layer, v2 keras layer has some bug

bn_mode = 'default' 
activ = None
kinit = tf.random_normal_initializer(0., 0.02)
if bn_mode== 'selu':
    kinit = tf.keras.initializers.LecunNormal()
    activ = 'selu'

def gen_layer(nfilts, ksize, input_shape, add_dropout=False):

    out = tf.keras.Sequential()
    out.add(tf.keras.layers.Conv2DTranspose(nfilts, ksize, strides=2, batch_input_shape=input_shape,
                                    padding='same',kernel_initializer=kinit,use_bias=False, activation = activ))
    
    
    if activ == 'selu':
        
        if add_dropout:
            out.add(tf.keras.layers.AlphaDropout(0.2))      
        
    if activ == None:
        out.add(BatchNormalization(momentum=0.8))  # Followed by a Batch Normalization Layer
        out.add(tf.keras.layers.ReLU())    
        if add_dropout:
            out.add(tf.keras.layers.Dropout(0.2))      # performing prunning / regularization

    return out



def makeGenerator():
    input_layer = tf.keras.layers.Input(shape=(100,))

    gen_stack = [
        tf.keras.layers.Dense(8*8*256, activation = activ),
        tf.keras.layers.Reshape((8,8,256)),
        gen_layer(128, 4, (None, 8, 8, 256), add_dropout=True), 
        gen_layer(64, 4, (None, 16, 16, 128), add_dropout=True), 
        gen_layer(32, 4, (None, 32, 32, 64), add_dropout=True),
        gen_layer(16, 4, (None, 64, 64, 32), add_dropout=True),
        
    ]

    last_layer = tf.keras.layers.Conv2DTranspose(3, 4,
                                           strides=1,
                                           padding='same',
                                           kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                           activation='tanh') # (BATCH_SIZE, 128, 128, 3)

    L = input_layer
    
    for layer in gen_stack:
        L = layer(L)
    
    L = last_layer(L)

    return tf.keras.Model(inputs=input_layer, outputs=L)