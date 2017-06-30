#-*- coding:utf-8 -*-
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
from featureProject.ly_features import get_actions
from sklearn.model_selection import train_test_split
from featureProject.ly_features import featureCombine
from featureProject.ly_features import featureCombine_2


######################### 训练数据 #########################
# data, labels = featureCombine_2()
data, labels = make_train_set(29000000, 30000000)

for i in range(18000000, 29000000, 1000000):
    tmp_data, tmp_labels = make_train_set(i,i+1000000)
    data = pd.concat([data,tmp_data])
    labels = pd.concat( [labels,tmp_labels] )

columns = data.columns

print "all samples:   %d*%d,pos/nag=%f"%(len(data.index),len(data.columns),
        1.0*len(labels[labels==1])/len(labels[labels==0]));

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
# test_data, test_labels_se = make_train_set(25000000,26000000)
################################## 评估数据 #########################################
# all_samples_train = lgb.Dataset(data.values, labels)
# deval = lgb.Dataset(X_test.values, y_test)
model = lgb.LGBMRegressor(
            objective='binary', 
            boosting_type='gbdt',
            num_leaves=300,
            max_depth= -1, 
            learning_rate=0.05,
            n_estimators=5000, 
            max_bin=255,
            subsample_for_bin=50000, # 构建箱的样本数
            min_split_gain=0,
            min_child_weight=1,
            min_child_samples=10, # 一个叶子上最少汇聚的样本数
            subsample=0.9, # 训练的随机样本数占比
            subsample_freq=1, # 样本子集的频率
            colsample_bytree=0.9,  # 每个投票人随机使用columns的占比
            #reg_alpha=0.1, # L1正则项系数
            #reg_lambda=0.1, # L2正则项系数
            seed=2017, 
            silent=True,
)
model.fit(
    X_train.values,  
    y_train, 
    eval_set=[ ( X_test.values, y_test ) ],
    eval_metric='binary_logloss',
    early_stopping_rounds=10,
)
# exit()

############################# 写记录到文件 ##########################################################
def get_xgb_feat_importances(clf):
    if isinstance(clf, xgb.XGBModel):
        # clf has been created by calling
        # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
        fscore = clf.booster().get_fscore()
    else:
        # clf has been created by calling xgb.train.
        # Thus, clf is an instance of xgb.Booster.
        fscore = clf.feature_importance()
    feat_importances = []
    for ft, score in fscore.iteritems():
        feat_importances.append({'Feature': ft, 'Importance': score})
    feat_importances = pd.DataFrame(feat_importances)
    feat_importances = feat_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    # Divide the importances by the sum of all importances
    # to get relative importances. By using relative importances
    # the sum of all importances will equal to 1, i.e.,
    # np.sum(feat_importances['importance']) == 1
    feat_importances['Importance'] /= feat_importances['Importance'].sum()
    # Print the most important features and their importances
    print feat_importances.head()
    return feat_importances
model = model.booster_
t = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
model.save_model('./model/model_' + t)


f = open('./featureImportance/feature_importance_' + t + '.txt', 'w')
f.write('features dimension : ' + str(len(columns)) + '\n\n')
# f.write('num_round : ' + str(num_boost_round) + '\n\n')
# f.write('test_loss : ' + str(cv_history['binary_logloss-mean'][-1]) + '\n\n')
f.write('feature importance : \n')
importances = model.feature_importance()
df = pd.DataFrame(importances, columns=['importance'])
df['feature_name'] = columns
df = df.sort_values(['importance'], ascending=False)
for i in range(0, len(df), 1):
    f.write(str(df.values[i][1]) + '\t'*5 +  str(df.values[i][0]) + '\n')
f.write('\n')
f.write('features :\n')
for i in range(len(columns)):
    f.write(columns[i] + ',\n')
f.write('\n')
f.close()

############################## 查看正确率 #############################################
# test_data, test_labels = make_train_set(25000000,26000000)
# y = model.predict( test_data )
# print report(test_labels, y)
############################ 生成提交文件 ###################################
test_data, test_labels = make_train_set(31000000, 32000000, sub=True)
instanceID = test_data['instanceID'].copy();
del test_data['instanceID']
print "sub samples:   %d*%d"%(len(test_data.index),len(test_data.columns))
y = model.predict(test_data)
pred = pd.concat([instanceID, pd.Series(y, name='prob')], axis=1)
pred = pred.sort_values('instanceID',ascending=True)
# fun = lambda x: 0.0 if x < 0 else x   # 为什么预测的还有负值
# pred['prob'] = pred['prob'].map(fun)
pred.to_csv('./sub/submission.csv', index=False, index_label=False)
