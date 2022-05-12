import random, os, sys
import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
import time

# from tqdm import tqdm
# from NeuralAgents.transformer_keras.dataloader import TokenList, pad_to_longest
from NeuralAgents.transformer_keras.transformer_keras import *

# from AutoCastSim.srunner.challenge.autoagents.transformer_keras.attention import *

import glob


# from https://github.com/Lsdefine/attention-is-all-you-need-keras


class Transformer_lidar:
    def __init__(self, input_dim, meta_size=2, n_action=3, batch_size=64, epoch=50,
                 d_model=256, # was 256
                 d_inner_hid=512, # was 512
                 n_head=4,
                 layers=2, # was 2
                 dropout=0.1):
        # params
        self.id = "Transformer_model"
        self._solver = 'adam'
        self.learning_rate = 0.00001 # was 0.001
        self._epoch = epoch
        self.batch_size = batch_size

        # self.i_tokens = i_tokens
        # self.o_tokens = o_tokens
        # self.len_limit = len_limit
        self.d_model = d_model
        self.decode_model = None
        self.readout_model = None
        self.layers = layers
        d_emb = d_model

        self.src_loc_info = True

        d_k = d_v = d_model // n_head
        assert d_k * n_head == d_model and d_v == d_k

        # self.pos_emb = PosEncodingLayer(len_limit, d_emb) if self.src_loc_info else None

        self.emb_dropout = Dropout(dropout)

        self.actor_num = input_dim[0]

        # self.input_emb = Embedding(self.size, d_emb)
        # self.meta_emb = Embedding(meta_size, d_emb)
        self.input_emb = Dense(d_emb)
        self.meta_emb = Dense(d_emb)

        self.encoder = SelfAttention(d_model, d_inner_hid, n_head, len_qkv=self.actor_num, layers=layers,  dropout=dropout)

        #self.encoder = MultiHeadSelfAttention(num_heads=n_head, use_masking=True, dropout=dropout)


        # self.decoder = Decoder(d_model, d_inner_hid, n_head, layers, dropout)
        # self.target_layer = TimeDistributed(Dense(o_tokens.num(), use_bias=False))
        self.target_layer = Dense(n_action)

        self.model = None
        self.compile(input_dim=input_dim, meta_size=meta_size, optimizer=self._solver)

        self.steps = 0

        filepath = self.id + ".ckpt"

        if os.path.exists(filepath):
            self.load_model(filepath)
            maxstep = 0
            for ckpt in glob.glob("*.ckpt_*"):
                maxstep = max(int(ckpt.split('_')[-1]), maxstep)
            self.steps = maxstep

    def compile(self, input_dim, meta_size, optimizer='adam', active_layers=999):

        # The mast input
        #mask_input = Input(shape=(280,280))#Input(shape=(1, self.actor_num))

        peer_meta_input = Input(shape=input_dim, name='peer')
        print("----Model Construction----")
        print("----peer_meta_input:", peer_meta_input.shape)
        # reshape to per actor data
        # src_input = Reshape((actor_num, input_dim[2]))(meta_input)
        # find per actor to ego embedding
        #(Jiaxun) Why timedistributed?
        src_input = Flatten()(peer_meta_input)
        src_emb = Dense(self.d_model)(src_input)
        src_emb = Dense(self.d_model)(src_emb)
        src_emb = Dense(128)(src_emb)
        src_emb = Dense(64)(src_emb)
        #src_emb = TimeDistributed(Dense(self.d_model))(peer_meta_input)
        #src_emb = TimeDistributed(Dense(self.d_model))(src_emb)
        #src_emb = self.emb_dropout(src_emb)
        enc_output=src_emb
        #enc_output = self.encoder(src_emb, mask_input, active_layers=active_layers, maskval=-1)

        # ego_output = enc_output[:,0,:]
        # print(enc_output.shape)
        #ego_output = Lambda(lambda x: x[:, 0, :])(enc_output)
        # ego_output = Flatten()(enc_output)
        ego_output=enc_output
        # enc_output = Flatten()(enc_output)
        # enc_output is a 3D matrix, of [batch size, input_dim, d_model], whereas Meta is a 2D matrix [batchsize, meta_dim]
        # enc_output = Dense(self.d_model)(enc_output) # TODO: is this the right thing to do?
        # ego_output = Reshape((1, self.d_model * self.actor_num))(enc_output)
        # # # enc_output = Reshape((1, self.d_model))(enc_output)
        # ego_output = Flatten()(ego_output)
        
        ego_meta_input =  Input(shape=(1,meta_size),name='meta') #Input(shape=(1, meta_size), name='meta')
        print("----ego_meta_input:", ego_meta_input.shape)

        meta_emb = Flatten()(ego_meta_input)
        meta_emb = Dense(self.d_model)(meta_emb)
        meta_emb = self.emb_dropout(meta_emb)
        meta_emb = Dense(self.d_model)(meta_emb)
        meta_emb = self.emb_dropout(meta_emb)

        concat_output = concatenate([ego_output, meta_emb])
        concat_output = Dense(self.d_model)(concat_output)
        concat_output = Dense(self.d_model)(concat_output)

        final_output = self.target_layer(concat_output)

        self.model = Model(inputs=[peer_meta_input, ego_meta_input], outputs=final_output)
        self.model.summary()

        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    def train(self, X_train, y_train, X_val, Y_val):
        # lr_scheduler = LRSchedulerPerStep(self.d_model, 4000)
        mfile = 'tmp.model.h5'
        model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

        K.set_value(self.model.optimizer.lr, self.learning_rate)
        print(self.model.optimizer.lr, self.learning_rate)
        t_start = time.time()
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self._epoch,
            verbose=1,
            validation_data=(X_val, Y_val),
            # callbacks=[model_saver]
        )
        t_end = time.time()
        print("Training Time: {}".format(t_end - t_start))

        # self.save(filepath)

        return history

    def predict(self, X_val):
        predicted = self.model.predict(X_val)
        return predicted

    def eval(self, X_val, val_y):
        # val_acc = self.model.evaluate(X_val, val_y, verbose=0)[1]
        val_loss = self.model.evaluate(X_val, val_y, verbose=1)[0]
        return val_loss

    def save(self, filepath):
        print("Saving Model...")
        self.model.save(filepath)

    def save_and_delete(self, filepath):
        print("Saving Model...")
        self.model.save(filepath)
        del self.model
        keras.backend.clear_session()

    def load_model(self, filepath):
        print("Loading Model...")

        self.model = load_model(
            filepath,
            custom_objects={
                # 'EmbeddingRet': EmbeddingRet,
                # 'TrigPosEmbedding': TrigPosEmbedding,
                #'MultiHeadAttention': MultiHeadAttention,
                #'MultiHeadSelfAttention': MultiHeadSelfAttention,
                #'LayerNormalization': LayerNormalization,
                # 'FeedForward': FeedForward,
                # 'EmbeddingSim': EmbeddingSim,
            })



