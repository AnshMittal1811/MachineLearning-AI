import argparse
import numpy as np
import tensorflow as tf
import logging
import os
from vaemols.utils import VAEDataGenerator
from vaemols.models import VariationalAutoencoder

parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--epochs', type=int, default=50, help='Training epochs')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0003, help='Learning rate of optimizer')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Minibatch size on training')
parser.add_argument('-ns', '--num_samples', type=int, default=4, help='Number to sample from latent distribution on training')
parser.add_argument('-ld', '--latent_dim', type=int, default=256, help='Latent space dimension')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data generator')
parser.add_argument('--max_length', type=int, default=120, help='Max length')
parser.add_argument('--save_model_dir', type=str, default='saved_models/', help='Model save dir')
parser.add_argument('--restore_model_file', type=str, help='Model file name to restore')

def main():
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    #load data
    x_train = np.load('data/x_train.npy')
    x_val_test = np.load('data/x_test.npy')
    x_val = x_val_test[:len(x_val_test)//2]
    x_test = x_val_test[len(x_val_test)//2:]
    logging.info('Number of train data: '+str(len(x_train)))
    logging.info('Number of validatoin data: '+str(len(x_val)))
    logging.info('Number of test data: '+str(len(x_test)))

    char_data = np.load('data/char_data.npz')
    int_to_char = char_data['int_to_char'].item()
    char_to_int = char_data['char_to_int'].item()
    num_classes = len(char_to_int)
    logging.info('Number of classes(char_set): '+str(num_classes))

    #data generator
    x_train_gen = VAEDataGenerator(x_train, num_classes, batch_size=args.batch_size)
    x_val_gen = VAEDataGenerator(x_val, num_classes, batch_size=args.batch_size)
    logging.info('Batch size: '+str(args.batch_size))

    #model
    inputs = tf.keras.layers.Input(shape=(args.max_length, num_classes))
    vae = VariationalAutoencoder(args.latent_dim, num_classes, args.max_length, num_samples=args.num_samples)
    outputs = vae(inputs)
    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate, clipvalue=0.1)
    vae.compile(optimizer=optimizer, loss=vae.vae_loss_func, metrics=[vae.sampled_data_acc])
    logging.info('Model compiled. Optimizer lr:'+str(args.learning_rate))

    #callbacks
    ckpt = tf.keras.callbacks.ModelCheckpoint(args.save_model_dir + 'weights-{epoch:02d}-{val_loss:.4f}.ckpt',
                                              monitor='val_loss',
                                              verbose=1, save_best_only=True,
                                              mode='auto', save_weights_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=0.0)

    #restore
    if args.restore_model_file is not None:
        restore_path = args.save_model_dir + args.restore_model_file
        vae.load_weights(restore_path)
        logging.info('Model restored from '+restore_path)

    #disable cuda logging message
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    vae.fit_generator(epochs=args.epochs, generator=x_train_gen, validation_data=x_val_gen,
                      use_multiprocessing=True, workers=args.num_workers,
                      callbacks=[ckpt, reduce_lr])

if __name__ == "__main__":
    main()
