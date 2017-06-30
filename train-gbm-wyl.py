#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import copy,os,sys
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import dump_svmlight_file
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif, f_regression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RandomizedLasso

from sklearn.model_selection import train_test_split
from featureProject.wyl_features import report
from featureProject.wyl_features import make_train_set
from featureProject.my_import import feature_importance2file
from featureProject.wyl_features import get_merge_train_data_set

models_usedfeats_path = "./featureSelect/models_usedfeats.csv"
feature_importance_file_path = "./logs/features_importance_xgb.csv"
train_days_seq = [ i for i in range( 28000000, 29000001, 1000000 ) ]
test_day = 25000000
#model_path = './model/wyl_gbm_49_28to30.model'
#model_path = './model/wyl_gbm_42_28to30.model'
model_path = './model/gbm_latest.model'
best_num = 38

def train_gbm_model(train_days_seq, test_day, model_path, best_num, model_type="cv",):
    models_usedfeats_df = pd.read_csv( models_usedfeats_path )
    best_feats = models_usedfeats_df[ models_usedfeats_df['gbm_feats']==1 ]['feat_name'].values

    if os.path.exists(model_path): os.remove(model_path); print "Remove file %s."%(model_path)
    if os.path.exists(model_path): 
        bst = lgb.Booster(model_file=model_path)
    else:
        ################################# 训练数据 ##########################################
        data, labels = get_merge_train_data_set(train_days_seq)
        data = data[ best_feats ]
        ################################## 评估数据 #########################################
        print "all samples:   %d*%d,pos/nag=%f"%(len(data.index),len(data.columns),1.0*len(labels[labels.label==1])/len(labels[labels.label==0]));
        ################################## 评估数据 #########################################
        X_train, X_test, y_train, y_test = train_test_split(data.values, labels['label'].values, test_size=0.15, random_state=0)
        all_samples_train = lgb.Dataset(data.values, labels['label'].values);
        deval = lgb.Dataset( X_test, y_test )
        columns = data.columns; data = None;
        params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'num_leaves': 300,
            'learning_rate': 0.05,
            'verbose': 0,
            'metric': {'binary_logloss'},
            #'device':'gpu',
        }
        num_boost_round = 130
        # if model_type == "cv":
        #     print "start cv, please wait ........"
        #     cv_history = lgb.cv(params, all_samples_train, num_boost_round, nfold=5, seed=2017, metrics = {"binary_logloss"}, early_stopping_rounds= 10, callbacks=[lgb.callback.print_evaluation(show_stdv=True)]);
        #     history_df = pd.DataFrame(cv_history)
        #     num_boost_round = len(history_df.index)
        # else: num_boost_round = 99 # 99
        bst = lgb.train(params, all_samples_train, num_boost_round, valid_sets = deval )
        bst.save_model(model_path)
        #feature_importance2file(bst, history_df, num_boost_round, feature_importance_file_path, columns, model_name='gbm')
    ############################## 查看正确率 #############################################
    test_data, test_labels_df = make_train_set(test_day,test_day+1000000)
    test_data = test_data[ best_feats ]
    print "all samples:   %d*%d."%(len(test_data.index),len(test_data.columns));
    y = bst.predict( test_data.values )
    report( test_labels_df['label'], y )
    test_labels_df, test_data = None, None #优化内存
    exit();
    ############################ 生成提交文件 ###################################
    test_data, test_labels = make_train_set(31000000, 32000000, sub=True)
    instanceID = test_data['instanceID'].copy(); del test_data['instanceID']
    test_data = test_data[ best_feats ]
    print "sub samples:   %d*%d"%(len(test_data.index),len(test_data.columns))
    y = bst.predict( test_data.values )
    pred = pd.concat([instanceID, pd.Series(y, name='prob')], axis=1)
    pred = pred.sort_values('instanceID',ascending=True)
    fun = lambda x: 0.0 if x < 0 else x   # 为什么预测的还有负值
    pred['prob'] = pred['prob'].map(fun)
    pred.to_csv('./sub/submission.csv', index=False, index_label=False)

if __name__ == '__main__':
    train_gbm_model(train_days_seq, test_day, model_path, best_num, model_type="cv")