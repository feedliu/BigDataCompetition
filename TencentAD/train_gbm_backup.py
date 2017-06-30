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
from featureProject.ly_features import make_train_set_with_basic_feat
from featureProject.ly_features import get_actions
from sklearn.model_selection import train_test_split
from featureProject.ly_features import featureCombine

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


######################### 训练数据 #########################
data, labels = make_train_set_with_basic_feat(24000000,25000000)
actions_before = get_actions(0, 24000000)
old_data = data[data['creativeID'].isin[actions_before['creativeID']]]
new_data = data[~data['creativeID'].isin[actions_before['creativeID']]]

for i in []:
    tmp_data, tmp_labels = make_train_set_with_basic_feat(i,i+1000000)
    data = pd.concat([data,tmp_data])
    labels = pd.concat( [labels,tmp_labels] )

data_index = data['creativeID'].isin(test_data['creativeID'].unique()) \
            & data['positionID'].isin(test_data['positionID'].unique()) \
            & data['appID'].isin(test_data['appID'].unique())

data = data[data_index]
labels = labels[data_index]

test_data, test_labels_se = make_train_set_with_basic_feat(25000000,26000000)


columns = data.columns

print "all samples:   %d*%d,pos/nag=%f"%(len(data.index),len(data.columns),1.0*len(labels[labels==1])/len(labels[labels==0]));
################################## 评估数据 #########################################
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
all_samples_train = lgb.Dataset(data.values, labels.values)
deval = lgb.Dataset(X_test.values, y_test)
params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'task': 'train',
    'num_leaves': 20,
    'max_depth':7,     
    'learning_rate': 0.05,
    'verbose': 0, 
    # 'n_estimators': 400, 
    'metric': {'binary_logloss'},
    # 'device':'gpu',
    'num_threads':7,
}
num_boost_round = 5000
print "start cv, please wait ........"
cv_history = lgb.cv(params, 
                    all_samples_train, 
                    num_boost_round, 
                    nfold=5, 
                    metrics = {"binary_logloss"}, 
                    early_stopping_rounds= 10, 
                    callbacks=[lgb.callback.print_evaluation(show_stdv=True)],
                    verbose_eval = False,
                    seed=2017,
                    )
# exit()
history_df = pd.DataFrame(cv_history)
num_boost_round = len(history_df.index)
bst = lgb.train(params, all_samples_train, num_boost_round, valid_sets=all_samples_train)

############################# 写记录到文件 ##########################################################
t = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
bst.save_model('./model/model_' + t)


f = open('./featureImportance/feature_importance_' + t + '.txt', 'w')
f.write('features dimension : ' + str(len(columns)) + '\n\n')
f.write('num_round : ' + str(num_boost_round) + '\n\n')
f.write('test_loss : ' + str(cv_history['binary_logloss-mean'][-1]) + '\n\n')
f.write('feature importance : \n')
importances = bst.feature_importance()
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
# test_data, test_labels_se = make_train_set_with_basic_feat(25000000,26000000)
y = bst.predict( test_data )
print report(test_labels_se, y)
test_labels_df, test_data = None, None #优化内存
exit();
############################ 生成提交文件 ###################################
# test_data, test_labels = make_train_set_with_basic_feat(31000000, 32000000, sub=True)
# instanceID = test_data['instanceID'].copy();
# del test_data['instanceID']
# print "sub samples:   %d*%d"%(len(test_data.index),len(test_data.columns))
# y = bst.predict(test_data, num_iteration=bst.best_iteration )
# pred = pd.concat([instanceID, pd.Series(y, name='prob')], axis=1)
# pred = pred.sort_values('instanceID',ascending=True)
# fun = lambda x: 0.0 if x < 0 else x   # 为什么预测的还有负值
# pred['prob'] = pred['prob'].map(fun)
# pred.to_csv('./sub/submission.csv', index=False, index_label=False)


nn_data = X_train[features] 
X_test = X_test[features]
nn_labels = y_train
def add_layer(inputs , input_feature_size , output_feature_size , activation_function = None):
    Weights = tf.Variable(tf.random_normal([input_feature_size , output_feature_size]))
    bias = tf.Variable(tf.zeros([1 , output_feature_size]) + 0.1)
    
    Wx_plus_bias = tf.matmul(inputs , Weights) + bias
    
    if(activation_function != None):
        outputs = activation_function(Wx_plus_bias)
    else :
        outputs = Wx_plus_bias
    
    return outputs

m, n = nn_data.shape
batch_size = 500
x_holder = tf.placeholder(dtype = np.float32 , shape = [None , n])
y_holder = tf.placeholder(dtype = np.float32 , shape = [None , ])

l1 = add_layer(x_holder , n , 4 , activation_function = tf.nn.sigmoid)
l2 = add_layer(l1 , 4 , 1 , activation_function = tf.nn.sigmoid)

epsilon = 1e-15
l3 = tf.maximum(epsilon, l2)
l4 = tf.minimum(1-epsilon, l3)

#define the loss
loss = tf.reduce_sum(-(y_holder * tf.log(l4) + (1 - y_holder) * tf.log(1 - l4)))

#define the proccess of train
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100000):
        start = i*batch_size % m if i*batch_size % m < m - batch_size else 0
        batch_x = nn_data.iloc[start:start + batch_size, :]
        batch_y = nn_labels[start:start + batch_size]
        sess.run(train , feed_dict={x_holder : batch_x, y_holder : batch_y})
        if(i % 50 == 0):
            print('step : %d, start : %d,\ttrain_loss : %f' % (i, start, sess.run(loss , feed_dict={x_holder : batch_x, y_holder : batch_y})))
            # print('eval_loss : %f' % sess.run(loss , feed_dict={x_holder : X_test, y_holder : y_test}))
