from keras.models import *
from keras.callbacks import *
import keras.backend as K
from model import *
import cv2
import argparse
import driving_data

def train():
        model = get_model()
        print "Loaded model"
        X, y = driving_data.get_validation_dataset()
        print model.summary()
        print "Loaded validation datasetset"
        print "Total of", len(y) * 4
        print "Training.."
        checkpoint_path="weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
        model.fit_generator(driving_data.generate_arrays_from_file(), validation_data = (X, y), samples_per_epoch = len(y) * 4, nb_epoch=150, verbose = 1, callbacks=[checkpoint])

if __name__ == "__main__":
    train()
