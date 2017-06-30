#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import xgboost as xgb
import time

from featureProject.ly_features import make_train_set
from featureProject.ly_features import onehot
from sklearn.model_selection import train_test_split
from featureProject.ly_features import report

################################### 读取模型 ###################################################
bst = None
# bst = xgb.Booster()
# bst.load_model('./model/model_2017-06-08_21:05:01')

################################### 读取数据 ##################################################
data, labels = make_train_set(24000000, 25000000)
for i in []:
	temp_data, temp_labels = make_train_set(i, i + 1000000) 
	data = pd.concat([data, temp_data])
	labels = pd.concat([labels, temp_labels])

columns = data.columns

# data = onehot(data, ['gender', 'education', 'mrariageStatus', 'haveBaby', 'connectionType'
#                                     , 'telecomsOperator', 'appPlatform', 'positionType'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
print "all samples:   %d*%d"%(len(data.index),len(data.columns));

# ################################## 数据划分 #########################################
dtrain = xgb.DMatrix(X_train.values,label = y_train, missing=-1)
deval = xgb.DMatrix(X_test.values, label = y_test, missing=-1)
all_train = xgb.DMatrix(data.values, label=labels, missing=-1)
del data, labels, X_train, X_test, y_train, y_test

#################################### 交叉验证 ################################################
params = { 'learning_rate' : 0.1,    'n_estimators': 1000,             'max_depth': 4, 
          'min_child_weight': 5,    'gamma': 0,                       'subsample': 1, 
          'colsample_bytree': 0.8,  'eta': 0.1, 'nthread' : 7, 'eval_metric' : 'logloss',
          'silent': 1,       'scale_pos_weight':1,     'objective': 'binary:logistic',
          }#'updater' : 'grow_gpu'
# num_round = 4000 
# cv_history = xgb.cv(params,
# 						all_train,
# 						num_round,
# 						nfold=5,
# 						metrics=['logloss'],
# 						early_stopping_rounds=10,
# 						seed=2017,
# 						callbacks=[xgb.callback.print_evaluation(show_stdv=True)],
# 						)
# exit()
################################ 训练模型 ################################################
# num_round = len(cv_history)
num_round = 50
plst = params.items()
evallist = [ (dtrain, 'train'), (deval, 'eval') ]
bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=10)
del all_train
time = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
bst.save_model('./model/xgb_model_' + time)

# ############################## 查看正确率 #############################################
test_data, test_labels = make_train_set(25000000, 26000000, test=True)
# data = onehot(data, ['gender', 'education', 'marriageStatus', 'haveBaby', 'connectionType'
#                                     , 'telecomsOperator', 'appPlatform', 'positionType'])

dtest = xgb.DMatrix(test_data.values, missing=-1)
y = bst.predict(dtest)

print report(test_labels, y)

############################ 生成提交文件 ###################################
# test_data, test_labels = make_train_set(31000000, 32000000, sub=True)
# instanceID = test_data['instanceID'].copy()
# del test_data['instanceID']
# dtest = xgb.DMatrix(test_data.values, missing=-1)
# y = bst.predict(dtest)
# pred = pd.concat([instanceID, pd.Series(y, name='prob')], axis=1)
# pred = pred.sort_values('instanceID',ascending=True)
# pred.to_csv('./sub/ly_submission_2017.csv', index=False, index_label=False)

def get_xgb_feat_importances(clf):
    if isinstance(clf, xgb.XGBModel):
        # clf has been created by calling
        # xgb.XGBClassifier.fit() or xgb.XGBRegressor().fit()
        fscore = clf.booster().get_fscore()
    else:
        # clf has been created by calling xgb.train.
        # Thus, clf is an instance of xgb.Booster.
        fscore = clf.get_fscore()
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

f = open('./featureImportance/feature_importance_' + time + '.txt', 'w')
f.write('features dimension : ' + str(len(columns)) + '\n\n')
f.write('num_round : ' + str(num_round) + '\n\n')
# f.write('test_loss : ' + str(cv_history.iloc[-1,0]) + '\n\n')
f.write('feature importance : \n')
importances = get_xgb_feat_importances(bst).values
count = 0
for item in importances[:, 0]:
	row = int(item.replace('f',''))
	f.write(str(columns[row]) + '\t'*5 +  str(importances[count, 1]) + '\n')
	count += 1
f.write('\n')
f.write('features :\n')
for i in range(len(columns)):
	f.write(columns[i] + ',\n')
f.write('\n')
f.close()