#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import xgboost as xgb
import copy,os,sys,psutil
import lightgbm as lgb
import time
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import dump_svmlight_file
# from svmutil import svm_read_problem

from featureProject.ly_features import report
from featureProject.ly_features import make_train_set_LR_feat
from featureProject.ly_features import make_train_set_with_basic_feat
from sklearn.linear_model import LogisticRegression 

######################### 训练数据 #########################
data, labels = make_train_set_LR_feat(24000000,25000000)


for i in []:
    tmp_data, tmp_labels = make_train_set_LR_feat(i,i+1000000)
    data = pd.concat([data,tmp_data])
    labels = pd.concat( [labels,tmp_labels] )

print "all samples:   %d*%d,pos/nag=%f"%(len(data.index),len(data.columns),1.0*len(labels[labels==1])/len(labels[labels==0]));
################################## 评估数据 #########################################
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

classifier = LogisticRegression(
                                C=0.01,
                                class_weight='balanced',
                                max_iter=1000,
                                random_state=2017,
                                tol=1e-4,
                                n_jobs=7,
                                )  # 使用类，参数全是默认的
classifier.fit(data.values, labels)  # 训练数据来学习，不需要返回值

y = classifier.predict([5, 3, 5, 2.5])  # 测试数据，分类返回标记


############################## 查看正确率 #############################################
test_data, test_labels_se = make_train_set_LR_feat(25000000,26000000)
y = classifier.predict(test_data.values)  # 测试数据，分类返回标记
print report(test_labels_se, y)
############################ 生成提交文件 ###################################
# test_data, test_labels = make_train_set_LR_feat(31000000, 32000000, sub=True)
# instanceID = test_data['instanceID'].copy();
# del test_data['instanceID']
# print "sub samples:   %d*%d"%(len(test_data.index),len(test_data.columns))
# y = bst.predict(test_data, num_iteration=bst.best_iteration )
# pred = pd.concat([instanceID, pd.Series(y, name='prob')], axis=1)
# pred = pred.sort_values('instanceID',ascending=True)
# fun = lambda x: 0.0 if x < 0 else x   # 为什么预测的还有负值
# pred['prob'] = pred['prob'].map(fun)
# pred.to_csv('./sub/submission.csv', index=False, index_label=False)
