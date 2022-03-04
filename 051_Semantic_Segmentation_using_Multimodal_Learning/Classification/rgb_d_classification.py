from keras import applications, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import convolutional, pooling, core, Input, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, ProgbarLogger

img_width, img_height = 256, 256
train_data_dir = "../Videos/rgb-dataset"
validation_data_dir = "../Videos/rgb-dataset"
train_data_dir_nir = "../Videos/d-dataset"
validation_data_dir_nir = "../Videos/d-dataset"
nb_train_samples = 207920
nb_validation_samples = 2079
batch_size = 10
epochs = 4
num_classes = 51


inp = Input(shape = (img_width , img_height, 3))
conv_layer1 = convolutional.Conv2D(8, (3,3), strides=(1, 1), padding='same', activation='relu')(inp)
conv_layer2 = convolutional.Conv2D(8, (3,3), strides=(1, 1), padding='same', activation='relu')(conv_layer1)
pool_layer1 = pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_layer2)
conv_layer3 = convolutional.Conv2D(16, (3,3), strides=(1, 1), padding='same', activation='relu')(pool_layer1)
pool_layer2 = pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_layer3)
conv_layer4 = convolutional.Conv2D(32, (3,3), strides=(1, 1), padding='same', activation='relu')(pool_layer2)
pool_layer3 = pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_layer4)
conv_layer5 = convolutional.Conv2D(32, (3,3), strides=(1, 1), padding='same', activation='relu')(conv_layer3)
pool_layer4 = pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_layer5)
flatten_layer = core.Flatten()(pool_layer4)
hidden1 = core.Dense(64, activation = 'relu')(flatten_layer)



inp_2 = Input(shape = (img_width , img_height, 3))
conv_layer1_2 = convolutional.Conv2D(8, (3,3), strides=(1, 1), padding='same', activation='relu')(inp_2)
conv_layer2_2 = convolutional.Conv2D(8, (3,3), strides=(1, 1), padding='same', activation='relu')(conv_layer1_2)
pool_layer1_2 = pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_layer2_2)
conv_layer3_2 = convolutional.Conv2D(16, (3,3), strides=(1, 1), padding='same', activation='relu')(pool_layer1_2)
pool_layer2_2 = pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_layer3_2)
conv_layer4_2 = convolutional.Conv2D(32, (3,3), strides=(1, 1), padding='same', activation='relu')(pool_layer2_2)
pool_layer3_2 = pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_layer4_2)
conv_layer5_2 = convolutional.Conv2D(32, (3,3), strides=(1, 1), padding='same', activation='relu')(conv_layer3_2)
pool_layer4_2 = pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_layer5_2)
flatten_layer_2 = core.Flatten()(pool_layer4_2)
hidden1_2 = core.Dense(64, activation = 'relu')(flatten_layer_2)

hidden_merge = concatenate([hidden1 , hidden1_2], axis=-1)

dropout1 = core.Dropout(0.2)(hidden_merge)
hidden2 = core.Dense(64,activation = 'relu')(dropout1)
out = core.Dense(num_classes,activation='softmax')(hidden2)

model1 = Model([inp, inp_2],out)

model1.compile(loss = "categorical_crossentropy", optimizer = 'rmsprop', metrics=["accuracy"])

seed = 7

def custom_iterator(Xp, Xs):
    from itertools import izip
    from keras.preprocessing.image import ImageDataGenerator

    ig1 = ImageDataGenerator(rescale=1./255)
    ig2 = ImageDataGenerator(rescale=1./255)
    temp1 = ig1.flow_from_directory(Xp,target_size = (img_height, img_width),batch_size = batch_size,class_mode = "categorical",seed=seed)
    temp2 = ig2.flow_from_directory(Xs,target_size = (img_height, img_width),batch_size = batch_size,class_mode = "categorical",seed=seed)


    for batch in izip(temp1,temp2):
        yield [batch[0][0], batch[1][0]], [batch[0][1]]


# Save the model according to the conditions  
progbar = ProgbarLogger(count_mode='steps')
checkpoint = ModelCheckpoint("rgbd.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=1, mode='auto')


# Train the model 
model1.fit_generator(
custom_iterator(train_data_dir_nir, train_data_dir),
steps_per_epoch = nb_train_samples,
epochs = epochs,
validation_data = custom_iterator(validation_data_dir_nir, validation_data_dir),
validation_steps = nb_validation_samples, 
callbacks = [progbar, checkpoint, early])
