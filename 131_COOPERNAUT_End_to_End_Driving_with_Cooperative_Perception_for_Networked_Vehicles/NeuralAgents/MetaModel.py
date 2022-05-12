import os
import glob
import math

import keras
from keras.layers import Input, concatenate
from keras.models import Model
import keras.backend as K
from keras.layers import Activation
from keras.layers import Dense,GlobalAveragePooling2D, BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import load_model


class MetaModel(object):
    def __init__(self, input_dim, n_action=3, batch_size=32, epoch=15):
        self.id = "MetaModel"
        self._epoch = epoch
        self._solver = 'Adam'
        self._learning_rate = 0.01
        self._lr_decay = 0.
        self._batch_size = batch_size
        self.model = self.build_model(input_dim, n_action)
        self.steps = 0

        filepath = self.id + ".ckpt"

        if os.path.exists(filepath):
            self.load_model(filepath)
            maxstep = 0
            for ckpt in glob.glob("*.ckpt_*"):
                maxstep = max(int(ckpt.split('_')[-1]), maxstep)
            self.steps = maxstep

    def build_model(self, input_shape, num_action=3):

        inputs = Input(shape=input_shape, name='meta')
        x = Flatten()(inputs)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # speed_input = Input(shape=(1, 1), name='spd')
        speed_input = Input(shape=(1, 2), name='spd')
        y = Flatten()(speed_input)
        y = Dense(64)(y)
        y = Activation('relu')(y)
        y = Dropout(0.5)(y)

        z = concatenate([x, y])
        z = Dense(128)(z)

        z = Dense(64)(z)

        outputs = Dense(num_action,
                        kernel_initializer='he_normal')(z)

        # Instantiate model.
        model = Model(inputs=[inputs, speed_input], outputs=outputs)

        # compile
        try:
            optimizer = getattr(keras.optimizers, self._solver)
        except:
            raise NotImplementedError('optimizer not implemented in keras')
        # All optimizers with the exception of nadam take decay as named arg
        try:
            opt = optimizer(lr=self._learning_rate, decay=self._lr_decay)
        except:
            raise NotImplementedError('optimizer error')

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])

        model.summary()

        return model

    def train(self, X, Y, X_val, Y_val):

        K.set_value(self.model.optimizer.lr, self._learning_rate)
        print("Training Actually started")
        # t_start = time.time()
        history = self.model.fit(
            X,
            Y,
            batch_size=self._batch_size,
            epochs=self._epoch,
            verbose=1,
            validation_data=(X_val, Y_val),
            # shuffle='batch'
        )
        # t_end = time.time()
        # print("Training Time: {}".format(t_end - t_start))
        return history

    def predict(self, input):
        return self.model.predict(input)

    def eval(self, X, Y):
        val_loss = self.model.evaluate(X, Y, verbose=1)[0]
        return val_loss

    def save(self, fp):
        print("Saving Model...")
        self.model.save(fp)

    def load_model(self, filepath):
        print("Loading Model...")
        self.model = load_model(filepath)

    def __del__(self):
        fp = "./{}.ckpt_emergency".format(self.id)
        self.save(fp)

