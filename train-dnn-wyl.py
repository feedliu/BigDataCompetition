#-*- coding:utf-8 -*-
import numpy as np
import copy,os,re,math
import scipy as sp
import pandas as pd

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

from featureProject.wyl_features import make_train_set
from sklearn.model_selection import train_test_split
from featureProject.wyl_features import report
from featureProject.wyl_features import get_merge_train_data_set


models_usedfeats_path = "./featureSelect/models_usedfeats.csv"
dnn_model_path = "./model/wyl_cnn_40.model"
train_days_seq = [ i for i in range( 29000000, 29000001, 1000000 ) ]
test_day = 25000000
best_num = 42;
pos_nag_ratio = 0.028082

def train_keras_model( train_x, train_y, dnn_model_path ):
	print train_x.shape,train_y.shape
	if os.path.exists(dnn_model_path): os.remove(dnn_model_path);
	if os.path.exists(dnn_model_path): model = load_model(dnn_model_path)
	else:
		model = Sequential()
		model.add( Dense(2, activation='sigmoid', input_shape=(train_x.shape[1],))  )
		#model.add( Dropout(0.25) )
		model.add( Dense(1, activation='sigmoid') )
		model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['acc'] )
		model.fit(
			train_x, 
			train_y, 
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
	return model

def train_dnn_model(train_days_seq, test_day, dnn_model_path, best_num, model_type="cv"):
	models_usedfeats_df = pd.read_csv( models_usedfeats_path )
	best_feats = models_usedfeats_df[ models_usedfeats_df['dnn_feats']==1 ]['feat_name'].values

	tr_data_df, tr_labels_df = get_merge_train_data_set(train_days_seq); #提取训练数据
	tr_data_df = tr_data_df[ best_feats ]
	model = train_keras_model( tr_data_df.values, tr_labels_df['label'], dnn_model_path,) #训练模型
	del tr_data_df; del tr_labels_df; #优化内存
	# ############################## 查看正确率 #############################################
	tt_data_df, tt_labels_df = make_train_set(test_day,test_day+1000000)
	tt_data_df = tt_data_df[ best_feats ]
	print "all samples:   %d*%d."%(len(tt_data_df.index),len(tt_data_df.columns));
	y = model.predict( tt_data_df.values )
	report( tt_labels_df['label'], y )
	print "max(y)=%f"%(max(y))
	tt_data_df, tt_labels_df = None, None #优化内存
	#exit();
	############################ 生成提交文件 ###################################
	test_data, test_labels = make_train_set(31000000, 32000000, sub=True)
	instanceID = test_data[['instanceID']].copy(); del test_data['instanceID']
	test_data = test_data[ best_feats ]
	print "sub samples:   %d*%d"%(len(test_data.index),len(test_data.columns))
	y = model.predict( test_data.values )
	pred = pd.concat([instanceID, pd.DataFrame(y,columns=['prob'])], axis=1)
	pred = pred.sort_values('instanceID',ascending=True)
	fun = lambda x: 0.0 if x < 0 else x   # 为什么预测的还有负值
	pred['prob'] = pred['prob'].map(fun)
	pred.to_csv('./sub/submission.csv', index=False, index_label=False)

if __name__ == '__main__':
	train_dnn_model(train_days_seq, test_day, dnn_model_path, best_num, model_type="cv")