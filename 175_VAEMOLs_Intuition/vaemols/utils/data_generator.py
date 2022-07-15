import tensorflow as tf
import numpy as np

class VAEDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, x, num_classes, batch_size=32):
        self.x = x
        self.num_classes = num_classes
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_one_hot = tf.keras.utils.to_categorical(batch_x, num_classes=self.num_classes)
        return (batch_x_one_hot, batch_x)
