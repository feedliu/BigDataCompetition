#-*- coding:utf-8 -*-
# import pandas as pd

# data = pd.read_csv('./data/train.csv')
# test = pd.read_csv('./data/test.csv')

# len(actions_19_20[actions_19_20['creativeID'].isin(test['creativeID'].unique()) \
# 			& actions_19_20['positionID'].isin(test['creativeID'].unique())])

import numpy as np
import pandas as pd
import scipy as sp
import xgboost as xgb
import copy,os,sys
import lightgbm as lgb
import time
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import dump_svmlight_file
# from svmutil import svm_read_problem

from featureProject.ly_features import report
from featureProject.ly_features import make_train_set
from featureProject.ly_features import make_train_set_with_basic_feat
from sklearn.model_selection import train_test_split
from featureProject.ly_features import featureCombine

######################### 训练数据 #########################
data, labels = make_train_set_with_basic_feat(24000000,25000000)


for i in range(17000000,30000000, 1000000):
    tmp_data, tmp_labels = make_train_set_with_basic_feat(i,i+1000000)
    data = pd.concat([data,tmp_data])
    labels = pd.concat( [labels,tmp_labels] )
