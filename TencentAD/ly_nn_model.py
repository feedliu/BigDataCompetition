#-*- coding:utf-8 -*-
import numpy as np
import copy,os,re,math
import scipy as sp
import pandas as pd
import time

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Embedding, LSTM
from keras.utils import np_utils
from keras.datasets import mnist
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.models import load_model
from tensorflow.contrib import learn
from sklearn import preprocessing

from featureProject.ly_features import make_train_set
from sklearn.model_selection import train_test_split
# from featureProject.ly_features import report

def report( right_se, pre_list ):
    epsilon = 1e-15
    act = np.array(right_se)
    ll = 0.0
    for i in range(len(pre_list)):
        pred = min( max(epsilon, pre_list[i]),1-  epsilon )
        ll += act[i] * math.log(pred) + (1.0 - act[i]) * math.log(1.0-pred)
    ll = ll * -1.0 / len(act)
    print 'my_loss = ',ll
    return ll

def sigmoid(X):
       return 1.0/(1+np.exp(-X))

t = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
dnn_model_path = "./model/ly_cnn_4.model_" + t
best_feats = ['positionID_appID_install_ratio', 'user_install_ratio', 'positionID_creativeID_install_ratio', 
            'connectionType_position_install_ratio', 'camgaignID_install_ratio', 'creative_click_install_ratio',
            'app_install_ratio', 'position_install_ratio', 'ad_install_ratio']

data, labels = make_train_set(24000000, 25000000)

for i in []:
    tmp_data, tmp_labels = make_train_set(i,i+1000000)
    data = pd.concat([data,tmp_data])
    labels = pd.concat( [labels,tmp_labels] )


test_data, test_labels = make_train_set(25000000, 26000000, sub=True)

data = data[ best_feats ]

model = Sequential()
model.add( Dense(4, activation='sigmoid', input_shape=(data.shape[1],))  )
#model.add( Dropout(0.25) )
model.add( Dense(1, activation='sigmoid') )
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['acc'] )
model.fit(
	data.values, 
	labels, 
	batch_size=256,
	epochs=50,
	verbose=1, 
	validation_split = 0.05,
	#class_weight = {1:1, 0:pos_nag_ratio},
	callbacks = [
		EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto'),
	],
)
model.save(dnn_model_path)

w = model.get_weights()
w1 = w[0]
b1 = w[1]
w2 = w[2]
b2 = w[3]

y = model.predict( test_data[best_feats].values )
print report(test_labels, y)