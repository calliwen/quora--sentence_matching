# coding:utf-8


# %matplotlib inline
# from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# ignore the warning: I tensorflow/core/platform/cpu_feature_guard.cc:140] 
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA


import tensorflow

import numpy as np
import pandas as pd
import datetime, time, json
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, dot, Flatten, Dense, Reshape, add, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import train_test_split

from keras.layers import Merge, Multiply, Concatenate

import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (numpy.asarray(self.model.predict(
            self.validation_data[:2]))).round()
        val_targ = self.validation_data[2]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return



class nn_model():
    def __init__(self):
        self.DATA_DIR = str("../data/")
        self.Q1_TRAINING_DATA_FILE = 'q1_train.npy'
        self.Q2_TRAINING_DATA_FILE = 'q2_train.npy'
        self.LABEL_TRAINING_DATA_FILE = 'label_train.npy'
        self.WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
        self.NB_WORDS_DATA_FILE = 'nb_words.json'
        self.MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
        
        self.MAX_SEQUENCE_LENGTH = 25
        self.WORD_EMBEDDING_DIM = 300
        
        self.SENT_EMBEDDING_DIM = 128

        self.VALIDATION_SPLIT = 0.1
        self.TEST_SPLIT = 0.1

        self.RNG_SEED = 42
        
        self.NB_EPOCHS = 50
        self.DROPOUT = 0.2
        self.BATCH_SIZE = 516
        self.get_keras_data()

    def get_keras_data(self):
        q1_data = np.load(open(self.DATA_DIR + self.Q1_TRAINING_DATA_FILE, 'rb'))
        q2_data = np.load(open(self.DATA_DIR + self.Q2_TRAINING_DATA_FILE, 'rb'))
        labels = np.load(open(self.DATA_DIR + self.LABEL_TRAINING_DATA_FILE, 'rb'))
        
        self.word_embedding_matrix = np.load(open( self.DATA_DIR + self.WORD_EMBEDDING_MATRIX_FILE, 'rb'))
        with open( self.DATA_DIR + self.NB_WORDS_DATA_FILE, 'r') as f:
            self.nb_words = json.load(f)['nb_words']
        
        X = np.stack((q1_data, q2_data), axis=1)
        y = labels
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, 
                                                                    test_size=self.TEST_SPLIT, 
                                                                    random_state=self.RNG_SEED,
                                                                    stratify=labels)
        self.Q1_train = X_train[:,0]
        self.Q2_train = X_train[:,1]
        self.Q1_test = X_test[:,0]
        self.Q2_test = X_test[:,1]
        print( "get data done !!! "  )


    def lstm_models(self, mode="cat"):
        self.mode = mode
        emb = Embedding(self.nb_words + 1, 
                        self.WORD_EMBEDDING_DIM, 
                        weights=[self.word_embedding_matrix], 
                        input_length=self.MAX_SEQUENCE_LENGTH, 
                        trainable=False)
        question_1 = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
        question_2 = Input(shape=(self.MAX_SEQUENCE_LENGTH,))

        q1 = LSTM( self.SENT_EMBEDDING_DIM )( emb(question_1) )
        q2 = LSTM( self.SENT_EMBEDDING_DIM )( emb(question_2) )
        
        if mode=='cat':
            res = Concatenate( axis=1 )( [q1, q2] )
            print( "concate.shape: ", res.get_shape()  )
        elif mode=='dis_agl':
            def Manhattan_distance(A,B):
                return K.sum( K.abs( A-B),axis=1,keepdims=True)
            merged_dist = Merge(mode=lambda x:Manhattan_distance(x[0],x[1]), 
                                  output_shape=lambda inp_shp:(inp_shp[0][0],1)  )([q1,q2])
            merged_agle = Multiply()( [q1, q2] )
            res = Concatenate( axis=1 )( [merged_dist, merged_agle] )

        res = Dense(128, activation='relu')(res)
        label = Dense(1, activation='sigmoid')(res)
        model = Model(inputs=[question_1,question_2], outputs=label)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print( model.summary() )
        return model

    def train_model(self, model):
        
        print("Starting training at", datetime.datetime.now())
        t0 = time.time()

        metrics = Metrics()
        callbacks = [metrics, ModelCheckpoint(self.MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
        
        
        history = model.fit([self.Q1_train, self.Q2_train],
                            self.y_train,
                            epochs=self.NB_EPOCHS,
                            # validation_split=self.VALIDATION_SPLIT,
                            validation_data=( [ self.Q1_test, self.Q2_test ], self.y_test ),
                            verbose=2,
                            batch_size=self.BATCH_SIZE,
                            callbacks=callbacks)
        t1 = time.time()
        print("Training ended at", datetime.datetime.now())
        print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

        acc = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                            'training': history.history['acc'],
                            'validation': history.history['val_acc'],
                            'val_f1': metrics.val_f1s
                            })
        acc.to_csv( "../data/res_history_{}.csv".format( self.mode ), index=False, encoding="utf-8" )

        # ax = acc.iloc[:,:].plot(x='epoch', figsize={5,8}, grid=True)
        # ax.set_ylabel("accuracy")
        # ax.set_ylim([0.0,1.0])
        pic = acc.plot( x='epoch', y=['training', 'validation', 'val_f1'], kind='line', grid=True )
        fig = pic.get_figure()
        fig.savefig( "../data/res_pic_{}.png".format( self.mode ) )
        
        max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
        print('Maximum accuracy at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(max_val_acc))

        model.load_weights(self.MODEL_WEIGHTS_FILE)
        loss, accuracy = model.evaluate([self.Q1_test, self.Q2_test], self.y_test, verbose=0)
        print('loss = {0:.4f}, accuracy = {1:.4f}'.format(loss, accuracy))



if __name__ =='__main__':
    nn_model = nn_model()
    lstm_model_1 = nn_model.lstm_models( mode='cat' )
    nn_model.train_model( lstm_model_1 )

    lstm_model_2 = nn_model.lstm_models( mode='dis_agl' )
    nn_model.train_model( lstm_model_2 )
