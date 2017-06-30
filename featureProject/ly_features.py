#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd
import os,copy,math,time,sys
import numpy as np
import scipy as sp

from sklearn.linear_model import RandomizedLasso

sys.path.append("../featureProject")


ad_path = './data/ad.csv'
app_categories_path = './data/app_categories.csv'
position_path = './data/position.csv'
test_path = './data/test.csv'
train_path = './data/train.csv'
user_path = './data/user.csv'
user_app_actions_path = './data/user_app_actions.csv'
user_installedapps_path = './data/user_installedapps.csv'

max_creativeID = 6582
max_adID = 3616
max_camgainID = 720
max_advertiserID = 91
max_appID=433269
max_appCategory = 503
max_positionID=7654
max_userID = 2805118
max_hometown = 3401
max_residence = 3401
max_installTime = 302359
max_clickTime = 302359
max_conversionTime = 302359

average_rate = 0.027
alpha = 100
beta = 3604

def bayes_smoothing(x, y):
    '''
    x : 分子
    y : 分母
    '''
    return (x + alpha) / (y + alpha + beta)

def time2seconds(x): #23232323
    # x = int('%o' % x, base=10)
    return x % 100 + x % 10000 // 100 * 60 + x % 1000000 // 10000 * 60 * 60 + x // 1000000 * 24 * 60 * 60


def time2minutes(x):  #23232323
    # x = int('%o' % x, base=10)
    return x % 10000 // 100 + x % 1000000 // 10000 * 60 + x // 1000000 * 24 * 60

def time2hours(x): #23232323
    # x = int('%o' % x, base=10)
    return  x // 10000 % 100 + x // 10000 // 100 * 24


def minutes2time(x):
    return x // (24 * 60) * 1000000 + x % (24 * 60) // 60 * 10000 + x % (24 * 60) % 60 * 100  

def age_map(x):
    if x == -1:
        return -1
    else:
        return x // 5

def appcate_map(x):
    second = x // 100
    if second == 0:
        return x * 100
    else:
        return x

def time_subtract_minute(x, delta):
    if x % 100 < delta: return x - 100 + 60 - delta
    else: return x - delta

def data_simulation(actions, start_date):
    actions.loc[actions['conversionTime'] >= start_date, 'label'] = 0
    return actions

def groupby_distributed(data, by_cols, compute_col, merge_cols, scale=5):
    i = int(len(data) / scale)
    new_data = data[0 : i]
    new_data = new_data.groupby(by_cols, as_index=False).count()

    for j in np.arange(2, scale):
        if j == scale: data_pro = data[(j - 1) * i : ]
        else: data_pro = data[(j - 1) * i : j * i]
        data_pro = data_pro.groupby(by_cols, as_index = False).count()

        new_data = pd.merge(new_data, data_pro, on=merge_cols, how='outer')
        new_data.fillna(0, inplace=True)
        new_data[compute_col] = new_data[compute_col + '_x'] + new_data[compute_col + '_y']
        del new_data[compute_col + '_x'], new_data[compute_col + '_y']

    return new_data

#读取整个用户基本的特征
def get_basic_user_feat():        # hometown , residence  are to be processed
    print('extract basic user feat...')
    dump_path = './cache/basic_user_feature.csv'
    feature = ['userID', 'age', 'gender', 'education', 'marriageStatus',
                'haveBaby'] #, 'hometown_first', 'hometown_second', 'residence_first', 'residence_second'
    if os.path.exists(dump_path): users = pd.read_csv(dump_path)
    else:
        users = pd.read_csv(user_path, encoding='gbk')
        # users['hometown_first'] = users['hometown'].map(lambda x : x // 100)
        # users['hometown_second'] = users['hometown'].map(lambda x : x % 100)
        # users['residence_first'] = users['residence'].map(lambda x : x // 100)
        # users['residence_second'] = users['residence'].map(lambda x : x % 100)
        users = users[feature]
        users.fillna(-1, inplace=True)
        users.replace(0, -1, inplace=True)  # replace the unknown value as -1
        # users.to_csv( dump_path, index=False, index_label=False )
    return users

# 读取广告素材的基本特征
def get_basic_ADcreative_feat():
    print('extract creative features...')
    dump_path = './cache/basic_ADcreative_feature.csv'
    if os.path.exists(dump_path): ads = pd.read_csv(dump_path)
    else:
        ads = pd.read_csv(ad_path)
    return ads

# 读取整个广告曝光位置的基本特征
def get_basic_position_feat():
    print('extract basic app positions...')
    dump_path = './cache/basic_position_feature.csv'
    if os.path.exists(dump_path): feat_df = pd.read_csv(dump_path)
    else:
        ad_positions = pd.read_csv(position_path)
        ad_positions['positionType'].fillna(-1, inplace=True)
        ad_positions['positionType'].replace(0, -1, inplace=True)
    return ad_positions

# 读取整个广告类别的基本特征
def get_basic_APPcategories_feat():
    print('extract basic APP categories...')
    dump_path = './cache/basic_APPcategories_feature.csv'
    if os.path.exists(dump_path): app_cates = pd.read_csv(dump_path)
    else:
        app_cates = pd.read_csv(app_categories_path)
        app_cates['appCategory'] = app_cates['appCategory'].map(appcate_map)
        app_cates['app_first_cate'] = app_cates['appCategory'].map(lambda x : x // 100)
        app_cates['app_second_cate'] = app_cates['appCategory'].map(lambda x : x % 100)
        del app_cates['appCategory']
        app_cates.fillna(-1, inplace=True)
        app_cates.replace(0, -1, inplace=True)
    return app_cates

def get_actions(start_date, end_date):
    print('extract actions from %d to %d...' % (start_date, end_date))
    dump_path = './cache/all_actions.csv'

    if os.path.exists(dump_path): actions = pd.read_csv(dump_path)
    else:      
        actions = pd.read_csv(train_path)
        actions = actions[ (actions['clickTime'] >= start_date) & (actions['clickTime'] < end_date) ]
        actions['connectionType'].fillna(-1, inplace=True)
        actions['connectionType'].replace(0, -1, inplace=True)
        actions['telecomsOperator'].fillna(-1, inplace=True)
        actions['telecomsOperator'].replace(0, -1, inplace=True)
        # actions.to_csv(dump_path,index=False, index_label=False)
    return actions

def get_test_actions(start_date, end_date):
    print('extract actions from %d to %d...' % (start_date, end_date))
    dump_path = './cache/all_test_actions.csv'

    if os.path.exists(dump_path): actions = pd.read_csv(dump_path)
    else:      
        actions = pd.read_csv(test_path)
        actions = actions[ (actions['clickTime'] >= start_date) & (actions['clickTime'] < end_date) ]
        actions['connectionType'].fillna(-1, inplace=True)
        actions['connectionType'].replace(0, -1, inplace=True)
        actions['telecomsOperator'].fillna(-1, inplace=True)
        actions['telecomsOperator'].replace(0, -1, inplace=True)
        # actions.to_csv(dump_path,index=False, index_label=False)
    return actions

def get_user_app_info_feat(start_date, end_date):
    print('extract user_app info features...')
    dump_path = './cache/user_app_info_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        app_actions = pd.read_csv(user_app_actions_path)
        app_actions = app_actions[(app_actions['installTime'] >= 1000000) & (app_actions['installTime'] < 17000000)]
        app_actions = app_actions[(app_actions['installTime'] >= start_date) & (app_actions['installTime'] < end_date)]
        app_actions = app_actions[['userID', 'appID']]

        features = app_actions

        train = pd.read_csv(train_path)
        train = data_simulation(train, end_date)
        train = train[train['label'] == 1]
        train = train[(train['clickTime'] >= start_date) & (train['clickTime'] < end_date)]
        train = train[['userID', 'creativeID']]
        creative = get_basic_ADcreative_feat()
        creative = creative[['creativeID', 'appID']]
        train = pd.merge(train, creative, on=['creativeID'], how='left')
        del creative
        train = train[['userID', 'appID']]

        features = pd.concat([features, train])
        
        if start_date <= 0:
            installed_app = pd.read_csv(user_installedapps_path)
            fetures = pd.concat([features, installed_app])

        features.to_csv(dump_path,index=False, index_label=False)
    return features

# the install ratio of the creative
def get_creative_install_ratio_feat(start_date, end_date):
    print('extract creative install ratio features...')
    dump_path = './cache/creative_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['creativeID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['creativeID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'creative_yes_num'}, inplace=True)

        features_click_num = features.groupby(['creativeID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'creative_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['creativeID'], how='left')

        features['creative_yes_num'].fillna(0, inplace=True)
        features['creative_click_num'].replace(0, np.nan, inplace=True)
        features['creative_click_install_ratio'] = features['creative_yes_num'] / features['creative_click_num']
        features['creative_click_install_ratio'].fillna(0, inplace=True)
        # features.fillna(0, inplace=True)
        # features['creative_click_install_ratio'] = bayes_smoothing(features['creative_yes_num'], features['creative_click_num'])


        features = features[['creativeID', 'creative_click_install_ratio', 'creative_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_app_install_ratio_feat(start_date, end_date):
    print('extract app install ratio features...')
    dump_path = './cache/app_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'app_yes_num'}, inplace=True)

        features_click_num = features.groupby(['appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'app_click_num'}, inplace=True)

        print('features merge...')
        features = pd.merge(features_click_num, features_click_yes, on=['appID'], how='left')

        features['app_yes_num'].fillna(0, inplace=True)
        features['app_click_num'].replace(0, np.nan, inplace=True)
        features['app_install_ratio'] = features['app_yes_num'] / features['app_click_num']
        features['app_install_ratio'].fillna(0, inplace=True)

        features = features[['appID', 'app_install_ratio', 'app_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_position_install_ratio_feat(start_date, end_date):
    print('extract position install ratio features...')
    dump_path = './cache/position_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['positionID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['positionID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'position_yes_num'}, inplace=True)

        features_click_num = features.groupby(['positionID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'position_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['positionID'], how='left')

        features['position_yes_num'].fillna(0, inplace=True)
        features['position_click_num'].replace(0, np.nan, inplace=True)
        features['position_install_ratio'] = features['position_yes_num'] / features['position_click_num']
        features['position_install_ratio'].fillna(0, inplace=True)

        features = features[['positionID', 'position_install_ratio', 'position_click_num',
                            'position_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_haveBaby_appID_install_ratio_feat(start_date, end_date):
    print('extract haveBaby app install ratio features...')
    dump_path = './cache/haveBaby_app_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['haveBaby', 'appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['haveBaby', 'appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'haveBaby_appID_yes_num'}, inplace=True)

        features_click_num = features.groupby(['haveBaby', 'appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'haveBaby_appID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['haveBaby', 'appID'], how='left')

        features['haveBaby_appID_yes_num'].fillna(0, inplace=True)
        features['haveBaby_appID_click_num'].replace(0, np.nan, inplace=True)
        features['haveBaby_appID_install_ratio'] = features['haveBaby_appID_yes_num'] / features['haveBaby_appID_click_num']
        features['haveBaby_appID_install_ratio'].fillna(0, inplace=True)

        features = features[['haveBaby', 'appID', 'haveBaby_appID_install_ratio', 'haveBaby_appID_click_num',
                                'haveBaby_appID_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_siteset_install_ratio_feat(start_date, end_date):
    print('extract siteset install ratio features...')
    dump_path = './cache/siteset_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_position_feat(), on=['positionID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['sitesetID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['sitesetID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['sitesetID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'siteset_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['sitesetID'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['siteset_click_num'].replace(0, np.nan, inplace=True)
        features['siteset_install_ratio'] = features['yes_num'] / features['siteset_click_num']
        features['siteset_install_ratio'].fillna(0, inplace=True)

        features = features[['sitesetID', 'siteset_install_ratio', 'siteset_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_camgaign_install_ratio_feat(start_date, end_date):
    print('extract camgaign install ratio features...')
    dump_path = './cache/camgaign_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['camgaignID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['camgaignID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['camgaignID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'camgaignID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['camgaignID'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['camgaignID_click_num'].replace(0, np.nan, inplace=True)
        features['camgaignID_install_ratio'] = features['yes_num'] / features['camgaignID_click_num']
        features['camgaignID_install_ratio'].fillna(0, inplace=True)

        features = features[['camgaignID', 'camgaignID_install_ratio', 'camgaignID_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_ad_install_ratio_feat(start_date, end_date):
    print('extract ad install ratio features...')
    dump_path = './cache/ad_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['adID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['adID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['adID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'adID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['adID'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['adID_click_num'].replace(0, np.nan, inplace=True)
        features['ad_install_ratio'] = features['yes_num'] / features['adID_click_num']
        features['ad_install_ratio'].fillna(0, inplace=True)

        features = features[['adID', 'ad_install_ratio', 'adID_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_advertise_install_ratio_feat(start_date, end_date):
    print('extract advertise install ratio features...')
    dump_path = './cache/advertise_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['advertiserID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['advertiserID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'advertiserID_yes_num'}, inplace=True)

        features_click_num = features.groupby(['advertiserID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'advertiserID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['advertiserID'], how='left')

        features['advertiserID_yes_num'].fillna(0, inplace=True)
        features['advertiserID_click_num'].replace(0, np.nan, inplace=True)
        features['advertiser_install_ratio'] = features['advertiserID_yes_num'] / features['advertiserID_click_num']
        features['advertiser_install_ratio'].fillna(0, inplace=True)

        features = features[['advertiserID', 'advertiser_install_ratio',
                            'advertiserID_click_num', 'advertiserID_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_connectType_install_ratio_feat(start_date, end_date):
    print('extract connectType install ratio features...')
    dump_path = './cache/connectType_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['connectionType', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['connectionType'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['connectionType'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'connectionType_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['connectionType'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['connectionType_click_num'].replace(0, np.nan, inplace=True)
        features['connectionType_install_ratio'] = features['yes_num'] / features['connectionType_click_num']
        features['connectionType_install_ratio'].fillna(0, inplace=True)

        features = features[['connectionType', 'connectionType_install_ratio', 'connectionType_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_connectType_creativeID_install_ratio_feat(start_date, end_date):
    print('extract connectType creative install ratio features...')
    dump_path = './cache/connectType_creative_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['connectionType', 'creativeID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['connectionType', 'creativeID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['connectionType', 'creativeID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'connectionType_creative_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['connectionType', 'creativeID'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['connectionType_creative_click_num'].replace(0, np.nan, inplace=True)
        features['connectionType_creative_install_ratio'] = features['yes_num'] / features['connectionType_creative_click_num']
        features['connectionType_creative_install_ratio'].fillna(0, inplace=True)

        features = features[['connectionType', 'creativeID', 'connectionType_creative_install_ratio', 'connectionType_creative_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_creative_education_install_ratio_feat(start_date, end_date):
    print('extract connectType creative install ratio features...')
    dump_path = './cache/connectType_creative_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['education', 'creativeID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['education', 'creativeID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'education_creative_yes_num'}, inplace=True)

        features_click_num = features.groupby(['education', 'creativeID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'education_creative_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['education', 'creativeID'], how='left')

        features['education_creative_yes_num'].fillna(0, inplace=True)
        features['education_creative_click_num'].replace(0, np.nan, inplace=True)
        features['education_creative_install_ratio'] = features['education_creative_yes_num'] / features['education_creative_click_num']
        features['education_creative_install_ratio'].fillna(0, inplace=True)

        features = features[['education', 'creativeID', 'education_creative_install_ratio',
                            'education_creative_click_num', 'education_creative_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_connectType_appID_install_ratio_feat(start_date, end_date):
    print('extract connectType app install ratio features...')
    dump_path = './cache/connectType_app_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['connectionType', 'appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['connectionType', 'appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'connectionType_APP_yes_num'}, inplace=True)

        features_click_num = features.groupby(['connectionType', 'appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'connectionType_appID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['connectionType', 'appID'], how='left')

        features['connectionType_APP_yes_num'].fillna(0, inplace=True)
        features['connectionType_appID_click_num'].replace(0, np.nan, inplace=True)
        features['connectionType_appID_install_ratio'] = features['connectionType_APP_yes_num'] / features['connectionType_appID_click_num']
        features['connectionType_appID_install_ratio'].fillna(0, inplace=True)

        features = features[['connectionType', 'appID', 'connectionType_appID_install_ratio',
                                'connectionType_appID_click_num', 'connectionType_APP_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_user_connectType_appID_install_ratio_feat(start_date, end_date):
    print('extract user connectType app install ratio features...')
    dump_path = './cache/user_connectType_app_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['userID' ,'connectionType', 'appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['userID' ,'connectionType', 'appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'user_connectionType_APP_yes_num'}, inplace=True)

        features_click_num = features.groupby(['userID' ,'connectionType', 'appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'user_connectionType_appID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['userID' ,'connectionType', 'appID'], how='left')

        features['user_connectionType_APP_yes_num'].fillna(0, inplace=True)
        features['user_connectionType_appID_click_num'].replace(0, np.nan, inplace=True)
        features['user_connectionType_appID_install_ratio'] = features['user_connectionType_APP_yes_num'] / features['user_connectionType_appID_click_num']
        features['user_connectionType_appID_install_ratio'].fillna(0, inplace=True)

        features = features[['userID' ,'connectionType', 'appID', 'user_connectionType_appID_install_ratio',
                                'user_connectionType_appID_click_num', 'user_connectionType_APP_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_gender_creativeID_install_ratio_feat(start_date, end_date):
    print('extract gender creative install ratio features...')
    dump_path = './cache/gender_creativeID_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['gender', 'creativeID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['gender', 'creativeID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['gender', 'creativeID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'gender_creative_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['gender', 'creativeID'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['gender_creative_click_num'].replace(0, np.nan, inplace=True)
        features['gender_creative_install_ratio'] = features['yes_num'] / features['gender_creative_click_num']
        features['gender_creative_install_ratio'].fillna(0, inplace=True)

        features = features[['gender', 'creativeID', 'gender_creative_install_ratio', 'gender_creative_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_gender_app_install_ratio_feat(start_date, end_date):
    print('extract gender app install ratio features...')
    dump_path = './cache/gender_app_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['gender', 'appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['gender', 'appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'gender_app_yes_num'}, inplace=True)

        features_click_num = features.groupby(['gender', 'appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'gender_appID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['gender', 'appID'], how='left')

        features['gender_app_yes_num'].fillna(0, inplace=True)
        features['gender_appID_click_num'].replace(0, np.nan, inplace=True)
        features['gender_appID_install_ratio'] = features['gender_app_yes_num'] / features['gender_appID_click_num']
        features['gender_appID_install_ratio'].fillna(0, inplace=True)

        features = features[['gender', 'appID', 'gender_appID_install_ratio', 'gender_appID_click_num',
                                'gender_app_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_position_app_install_ratio_feat(start_date, end_date):
    print('extract positionID app install ratio features...')
    dump_path = './cache/positionID_app_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['positionID', 'appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['positionID', 'appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'positionID_app_yes_num'}, inplace=True)

        features_click_num = features.groupby(['positionID', 'appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'positionID_appID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['positionID', 'appID'], how='left')

        features['positionID_app_yes_num'].fillna(0, inplace=True)
        features['positionID_appID_click_num'].replace(0, np.nan, inplace=True)
        features['positionID_appID_install_ratio'] = features['positionID_app_yes_num'] / features['positionID_appID_click_num']
        features['positionID_appID_install_ratio'].fillna(0, inplace=True)

        features = features[['positionID', 'appID', 'positionID_appID_install_ratio', 'positionID_appID_click_num',
                                'positionID_app_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_position_creative_install_ratio_feat(start_date, end_date):
    print('extract positionID creative install ratio features...')
    dump_path = './cache/positionID_creative_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        # features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['positionID', 'creativeID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['positionID', 'creativeID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'positionID_creativeID_yes_num'}, inplace=True)

        features_click_num = features.groupby(['positionID', 'creativeID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'positionID_creativeID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['positionID', 'creativeID'], how='left')

        features['positionID_creativeID_yes_num'].fillna(0, inplace=True)
        features['positionID_creativeID_click_num'].replace(0, np.nan, inplace=True)
        features['positionID_creativeID_install_ratio'] = features['positionID_creativeID_yes_num'] / features['positionID_creativeID_click_num']
        features['positionID_creativeID_install_ratio'].fillna(0, inplace=True)

        features = features[['positionID', 'creativeID', 'positionID_creativeID_install_ratio', 'positionID_creativeID_click_num',
                                'positionID_creativeID_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_position_ad_install_ratio_feat(start_date, end_date):
    print('extract positionID ad install ratio features...')
    dump_path = './cache/positionID_ad_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['positionID', 'adID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['positionID', 'adID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'positionID_ad_yes_num'}, inplace=True)

        features_click_num = features.groupby(['positionID', 'adID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'positionID_ad_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['positionID', 'adID'], how='left')

        features['positionID_ad_yes_num'].fillna(0, inplace=True)
        features['positionID_ad_click_num'].replace(0, np.nan, inplace=True)
        features['positionID_ad_install_ratio'] = features['positionID_ad_yes_num'] / features['positionID_ad_click_num']
        features['positionID_ad_install_ratio'].fillna(0, inplace=True)

        features = features[['positionID', 'adID', 'positionID_ad_install_ratio', 'positionID_ad_click_num',
                                'positionID_ad_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_position_campaignID_install_ratio_feat(start_date, end_date):
    print('extract positionID campaignID install ratio features...')
    dump_path = './cache/positionID_campaignID_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['positionID', 'camgaignID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['positionID', 'camgaignID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'positionID_campaignID_yes_num'}, inplace=True)

        features_click_num = features.groupby(['positionID', 'camgaignID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'positionID_campaignID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['positionID', 'camgaignID'], how='left')

        features['positionID_campaignID_yes_num'].fillna(0, inplace=True)
        features['positionID_campaignID_click_num'].replace(0, np.nan, inplace=True)
        features['positionID_campaignID_install_ratio'] = features['positionID_campaignID_yes_num'] / features['positionID_campaignID_click_num']
        features['positionID_campaignID_install_ratio'].fillna(0, inplace=True)

        features = features[['positionID', 'camgaignID', 'positionID_campaignID_install_ratio', 'positionID_campaignID_click_num',
                                'positionID_campaignID_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_position_advertiseID_install_ratio_feat(start_date, end_date):
    print('extract positionID advertiseID install ratio features...')
    dump_path = './cache/positionID_advertiseID_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['positionID', 'advertiserID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['positionID','advertiserID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'positionID_advertiseID_yes_num'}, inplace=True)

        features_click_num = features.groupby(['positionID', 'advertiserID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'positionID_advertiseID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['positionID', 'advertiserID'], how='left')

        features['positionID_advertiseID_yes_num'].fillna(0, inplace=True)
        features['positionID_advertiseID_click_num'].replace(0, np.nan, inplace=True)
        features['positionID_advertiseID_install_ratio'] = features['positionID_advertiseID_yes_num'] / features['positionID_advertiseID_click_num']
        features['positionID_advertiseID_install_ratio'].fillna(0, inplace=True)

        features = features[['positionID', 'advertiserID', 'positionID_advertiseID_install_ratio', 'positionID_advertiseID_click_num',
                                'positionID_advertiseID_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_connectionType_position_app_install_ratio_feat(start_date, end_date):
    print('extract connectionType positionID app install ratio features...')
    dump_path = './cache/connectionType_positionID_app_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['connectionType', 'positionID', 'appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['connectionType', 'positionID', 'appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'connectionType_positionID_app_yes_num'}, inplace=True)

        features_click_num = features.groupby(['connectionType', 'positionID', 'appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'connectionType_positionID_appID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['connectionType', 'positionID', 'appID'], how='left')

        features['connectionType_positionID_app_yes_num'].fillna(0, inplace=True)
        features['connectionType_positionID_appID_click_num'].replace(0, np.nan, inplace=True)
        features['connectionType_positionID_appID_install_ratio'] = features['connectionType_positionID_app_yes_num'] / features['connectionType_positionID_appID_click_num']
        features['connectionType_positionID_appID_install_ratio'].fillna(0, inplace=True)

        features = features[['connectionType', 'positionID', 'appID', 'connectionType_positionID_appID_install_ratio', 'connectionType_positionID_appID_click_num',
                                'connectionType_positionID_app_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_user_position_app_install_ratio_feat(start_date, end_date):
    print('extract user positionID app install ratio features...')
    dump_path = './cache/user_positionID_app_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['userID','positionID', 'appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['userID','positionID', 'appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'user_positionID_app_yes_num'}, inplace=True)

        features_click_num = features.groupby(['userID','positionID', 'appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'user_positionID_appID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['userID','positionID', 'appID'], how='left')

        features['user_positionID_app_yes_num'].fillna(0, inplace=True)
        features['user_positionID_appID_click_num'].replace(0, np.nan, inplace=True)
        features['user_positionID_appID_install_ratio'] = features['user_positionID_app_yes_num'] / features['user_positionID_appID_click_num']
        features['user_positionID_appID_install_ratio'].fillna(0, inplace=True)

        features = features[['userID','positionID', 'appID', 'user_positionID_appID_install_ratio', 'user_positionID_appID_click_num',
                                'user_positionID_app_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_install_ratio_feat(start_date, end_date):
    print('extract age install ratio features...')
    dump_path = './cache/age_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['age', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_yes_num'}, inplace=True)

        features_click_num = features.groupby(['age'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age'], how='left')

        features['age_yes_num'].fillna(0, inplace=True)
        features['age_click_num'].replace(0, np.nan, inplace=True)
        features['age_install_ratio'] = features['age_yes_num'] / features['age_click_num']
        features['age_install_ratio'].fillna(0, inplace=True)

        features = features[['age', 'age_install_ratio', 'age_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_gender_creativeID_install_ratio_feat(start_date, end_date):
    print('extract age gender creative install ratio features...')
    dump_path = './cache/age_gender_creativeID_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['gender', 'age', 'creativeID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age','gender', 'creativeID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['age','gender', 'creativeID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_gender_creative_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age','gender', 'creativeID'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['age_gender_creative_click_num'].replace(0, np.nan, inplace=True)
        features['age_gender_creative_install_ratio'] = features['yes_num'] / features['age_gender_creative_click_num']
        features['age_gender_creative_install_ratio'].fillna(0, inplace=True)

        features = features[['age','gender', 'creativeID', 'age_gender_creative_install_ratio', 'age_gender_creative_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_gender_appID_install_ratio_feat(start_date, end_date):
    print('extract age gender appID install ratio features...')
    dump_path = './cache/age_gender_appID_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['gender', 'age', 'appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age','gender', 'appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_gender_appID_yes_num'}, inplace=True)

        features_click_num = features.groupby(['age','gender', 'appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_gender_appID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age','gender', 'appID'], how='left')

        features['age_gender_appID_yes_num'].fillna(0, inplace=True)
        features['age_gender_appID_click_num'].replace(0, np.nan, inplace=True)
        features['age_gender_appID_install_ratio'] = features['age_gender_appID_yes_num'] / features['age_gender_appID_click_num']
        features['age_gender_appID_install_ratio'].fillna(0, inplace=True)

        features = features[['age','gender', 'appID', 'age_gender_appID_install_ratio',
                            'age_gender_appID_click_num', 'age_gender_appID_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_creative_install_ratio_feat(start_date, end_date):
    print('extract age_creative install ratio features...')
    dump_path = './cache/age_creative_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['age', 'creativeID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age', 'creativeID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['age', 'creativeID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_creative_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age', 'creativeID'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['age_creative_click_num'].replace(0, np.nan, inplace=True)
        features['age_creative_install_ratio'] = features['yes_num'] / features['age_creative_click_num']
        features['age_creative_install_ratio'].fillna(0, inplace=True)

        features = features[['age', 'creativeID', 'age_creative_install_ratio', 'age_creative_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_app_install_ratio_feat(start_date, end_date):
    print('extract age_app install ratio features...')
    dump_path = './cache/age_app_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['age', 'appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age', 'appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_appID_yes_num'}, inplace=True)

        features_click_num = features.groupby(['age', 'appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_appID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age', 'appID'], how='left')

        features['age_appID_yes_num'].fillna(0, inplace=True)
        features['age_appID_click_num'].replace(0, np.nan, inplace=True)
        features['age_appID_install_ratio'] = features['age_appID_yes_num'] / features['age_appID_click_num']
        features['age_appID_install_ratio'].fillna(0, inplace=True)

        features = features[['age', 'appID', 'age_appID_install_ratio', 'age_appID_click_num', 'age_appID_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_camgaignID_install_ratio_feat(start_date, end_date):
    print('extract age_camgaignID install ratio features...')
    dump_path = './cache/age_camgaignID_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['age', 'camgaignID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age', 'camgaignID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_camgaignID_yes_num'}, inplace=True)

        features_click_num = features.groupby(['age', 'camgaignID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_camgaignIDID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age', 'camgaignID'], how='left')

        features['age_camgaignID_yes_num'].fillna(0, inplace=True)
        features['age_camgaignIDID_click_num'].replace(0, np.nan, inplace=True)
        features['age_camgaignID_install_ratio'] = features['age_camgaignID_yes_num'] / features['age_camgaignIDID_click_num']
        features['age_camgaignID_install_ratio'].fillna(0, inplace=True)

        features = features[['age', 'camgaignID', 'age_camgaignID_install_ratio',
                            'age_camgaignIDID_click_num', 'age_camgaignID_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_position_install_ratio_feat(start_date, end_date):
    print('extract age_position install ratio features...')
    dump_path = './cache/age_position_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['age', 'positionID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age', 'positionID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_position_yes_num'}, inplace=True)

        features_click_num = features.groupby(['age', 'positionID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_position_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age', 'positionID'], how='left')

        features['age_position_yes_num'].fillna(0, inplace=True)
        features['age_position_click_num'].replace(0, np.nan, inplace=True)
        features['age_position_install_ratio'] = features['age_position_yes_num'] / features['age_position_click_num']
        features['age_position_install_ratio'].fillna(0, inplace=True)

        features = features[['age', 'positionID', 'age_position_install_ratio',
                            'age_position_click_num', 'age_position_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_cate_install_ratio_feat(start_date, end_date):
    print('extract age_cate install ratio features...')
    dump_path = './cache/age_cate_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = pd.merge(features, get_basic_APPcategories_feat(), on=['appID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['age', 'app_first_cate', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age', 'app_first_cate'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_cate_yes_num'}, inplace=True)

        features_click_num = features.groupby(['age', 'app_first_cate'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_cate_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age', 'app_first_cate'], how='left')

        features['age_cate_yes_num'].fillna(0, inplace=True)
        features['age_cate_click_num'].replace(0, np.nan, inplace=True)
        features['age_cate_install_ratio'] = features['age_cate_yes_num'] / features['age_cate_click_num']
        features['age_cate_install_ratio'].fillna(0, inplace=True)

        features = features[['age', 'app_first_cate', 'age_cate_install_ratio', 'age_cate_click_num',
                                'age_cate_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_positionType_install_ration_feat(start_date, end_date):
    print('extract age_positionType install ratio features...')
    dump_path = './cache/age_positionType_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_position_feat(), on=['positionID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['age', 'positionType', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age', 'positionType'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_position_yes_num'}, inplace=True)

        features_click_num = features.groupby(['age', 'positionType'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_positionType_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age', 'positionType'], how='left')

        features['age_position_yes_num'].fillna(0, inplace=True)
        features['age_positionType_click_num'].replace(0, np.nan, inplace=True)
        features['age_positionType_install_ratio'] = features['age_position_yes_num'] / features['age_positionType_click_num']
        features['age_positionType_install_ratio'].fillna(0, inplace=True)

        features = features[['age', 'positionType', 'age_position_yes_num',
                                'age_positionType_click_num', 'age_positionType_install_ratio']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_education_install_ration_feat(start_date, end_date):
    print('extract age_education install ratio features...')
    dump_path = './cache/age_education_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_position_feat(), on=['positionID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['age', 'education', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age', 'education'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_education_yes_num'}, inplace=True)

        features_click_num = features.groupby(['age', 'education'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_education_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age', 'education'], how='left')

        features['age_education_yes_num'].fillna(0, inplace=True)
        features['age_education_click_num'].replace(0, np.nan, inplace=True)
        features['age_education_install_ratio'] = features['age_education_yes_num'] / features['age_education_click_num']
        features['age_education_install_ratio'].fillna(0, inplace=True)

        features = features[['age', 'education', 'age_education_yes_num',
                                'age_education_click_num', 'age_education_install_ratio']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_hour_install_ration_feat(start_date, end_date):
    print('extract age_hour install ratio features...')
    dump_path = './cache/age_hour_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_position_feat(), on=['positionID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features['clickHour'] = features['clickTime'].map(lambda x : x // 10000 % 100)
        features = features[['age', 'clickHour', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age', 'clickHour'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_clickHour_yes_num'}, inplace=True)

        features_click_num = features.groupby(['age', 'clickHour'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_clickHour_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age', 'clickHour'], how='left')

        features['age_clickHour_yes_num'].fillna(0, inplace=True)
        features['age_clickHour_click_num'].replace(0, np.nan, inplace=True)
        features['age_clickHour_install_ratio'] = features['age_clickHour_yes_num'] / features['age_clickHour_click_num']
        features['age_clickHour_install_ratio'].fillna(0, inplace=True)

        features = features[['age', 'clickHour', 'age_clickHour_yes_num',
                                'age_clickHour_click_num', 'age_clickHour_install_ratio']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_education_hour_install_ration_feat(start_date, end_date):
    print('extract education_hour install ratio features...')
    dump_path = './cache/education_hour_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_position_feat(), on=['positionID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features['clickHour'] = features['clickTime'].map(lambda x : x // 10000 % 100)
        features = features[['education', 'clickHour', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['education', 'clickHour'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'education_clickHour_yes_num'}, inplace=True)

        features_click_num = features.groupby(['education', 'clickHour'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'education_clickHour_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['education', 'clickHour'], how='left')

        features['education_clickHour_yes_num'].fillna(0, inplace=True)
        features['education_clickHour_click_num'].replace(0, np.nan, inplace=True)
        features['education_clickHour_install_ratio'] = features['education_clickHour_yes_num'] / features['education_clickHour_click_num']
        features['education_clickHour_install_ratio'].fillna(0, inplace=True)

        features = features[['education', 'clickHour', 'education_clickHour_yes_num',
                                'education_clickHour_click_num', 'education_clickHour_install_ratio']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_creative_hour_install_ration_feat(start_date, end_date):
    print('extract creative_hour install ratio features...')
    dump_path = './cache/creative_hour_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_position_feat(), on=['positionID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features['clickHour'] = features['clickTime'].map(lambda x : x // 10000 % 100)
        features = features[['creativeID', 'clickHour', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['creativeID', 'clickHour'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'creativeID_clickHour_yes_num'}, inplace=True)

        features_click_num = features.groupby(['creativeID', 'clickHour'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'creativeID_clickHour_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['creativeID', 'clickHour'], how='left')

        features['creativeID_clickHour_yes_num'].fillna(0, inplace=True)
        features['creativeID_clickHour_click_num'].replace(0, np.nan, inplace=True)
        features['creativeID_clickHour_install_ratio'] = features['creativeID_clickHour_yes_num'] / features['creativeID_clickHour_click_num']
        features['creativeID_clickHour_install_ratio'].fillna(0, inplace=True)

        features = features[['creativeID', 'clickHour', 'creativeID_clickHour_yes_num',
                                'creativeID_clickHour_click_num', 'creativeID_clickHour_install_ratio']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_telecomsOperator_install_ration_feat(start_date, end_date):
    print('extract age_telecomsOperator install ratio features...')
    dump_path = './cache/age_telecomsOperator_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_position_feat(), on=['positionID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['age', 'telecomsOperator', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age', 'telecomsOperator'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_telecomsOperator_yes_num'}, inplace=True)

        features_click_num = features.groupby(['age', 'telecomsOperator'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_telecomsOperator_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age', 'telecomsOperator'], how='left')

        features['age_telecomsOperator_yes_num'].fillna(0, inplace=True)
        features['age_telecomsOperator_click_num'].replace(0, np.nan, inplace=True)
        features['age_telecomsOperator_install_ratio'] = features['age_telecomsOperator_yes_num'] / features['age_telecomsOperator_click_num']
        features['age_telecomsOperator_install_ratio'].fillna(0, inplace=True)

        features = features[['age', 'telecomsOperator', 'age_telecomsOperator_yes_num',
                                'age_telecomsOperator_click_num', 'age_telecomsOperator_install_ratio']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_first_cate_install_ration_feat(start_date, end_date):
    print('extract age_first_cate install ratio features...')
    dump_path = './cache/age_first_cate_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = pd.merge(features, get_basic_APPcategories_feat(), on=['appID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['age', 'app_first_cate', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age', 'app_first_cate'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_app_first_cate_yes_num'}, inplace=True)

        features_click_num = features.groupby(['age', 'app_first_cate'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_app_first_cate_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age', 'app_first_cate'], how='left')

        features['age_app_first_cate_yes_num'].fillna(0, inplace=True)
        features['age_app_first_cate_click_num'].replace(0, np.nan, inplace=True)
        features['age_app_first_cate_install_ratio'] = features['age_app_first_cate_yes_num'] / features['age_app_first_cate_click_num']
        features['age_app_first_cate_install_ratio'].fillna(0, inplace=True)

        features = features[['age', 'app_first_cate', 'age_app_first_cate_yes_num',
                                'age_app_first_cate_click_num', 'age_app_first_cate_install_ratio']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_age_connectionType_install_ratio_feat(start_date, end_date):
    print('extract age_connectionType install ratio features...')
    dump_path = './cache/age_connectionType_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['age', 'connectionType', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['age', 'connectionType'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_connectionType_yes_num'}, inplace=True)

        features_click_num = features.groupby(['age', 'connectionType'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'age_connectionType_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['age', 'connectionType'], how='left')

        features['age_connectionType_yes_num'].fillna(0, inplace=True)
        features['age_connectionType_click_num'].replace(0, np.nan, inplace=True)
        features['age_connectionType_install_ratio'] = features['age_connectionType_yes_num'] / features['age_connectionType_click_num']
        features['age_connectionType_install_ratio'].fillna(0, inplace=True)

        features = features[['age', 'connectionType', 'age_connectionType_install_ratio', 'age_connectionType_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_education_connectionType_install_ratio_feat(start_date, end_date):
    print('extract education_connectionType install ratio features...')
    dump_path = './cache/education_connectionType_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['education', 'connectionType', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['education', 'connectionType'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'age_connectionType_yes_num'}, inplace=True)

        features_click_num = features.groupby(['education', 'connectionType'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'education_connectionType_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['education', 'connectionType'], how='left')

        features['age_connectionType_yes_num'].fillna(0, inplace=True)
        features['education_connectionType_click_num'].replace(0, np.nan, inplace=True)
        features['education_connectionType_install_ratio'] = features['age_connectionType_yes_num'] / features['education_connectionType_click_num']
        features['education_connectionType_install_ratio'].fillna(0, inplace=True)

        features = features[['education', 'connectionType', 'education_connectionType_install_ratio', 'education_connectionType_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_app_first_cate_position_install_ratio_feat(start_date, end_date):
    print('extract first_cate_position install ratio features...')
    dump_path = './cache/app_first_cate_position_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = pd.merge(features, get_basic_APPcategories_feat(), on=['appID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['app_first_cate', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['app_first_cate'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['app_first_cate'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'app_first_cate_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['app_first_cate'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['app_first_cate_click_num'].replace(0, np.nan, inplace=True)
        features['app_first_cate_install_ratio'] = features['yes_num'] / features['app_first_cate_click_num']
        features['app_first_cate_install_ratio'].fillna(0, inplace=True)

        features = features[['app_first_cate', 'app_first_cate_install_ratio', 'app_first_cate_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_app_first_cate_creative_install_ratio_feat(start_date, end_date):
    print('extract first_cate_creative install ratio features...')
    dump_path = './cache/app_first_cate_creative_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = pd.merge(features, get_basic_APPcategories_feat(), on=['appID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['app_first_cate', 'creativeID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['app_first_cate', 'creativeID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['app_first_cate', 'creativeID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'app_first_cate_creative_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['app_first_cate', 'creativeID'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['app_first_cate_creative_click_num'].replace(0, np.nan, inplace=True)
        features['app_first_cate_creative_install_ratio'] = features['yes_num'] / features['app_first_cate_creative_click_num']
        features['app_first_cate_creative_install_ratio'].fillna(0, inplace=True)

        features = features[['app_first_cate', 'creativeID', 'app_first_cate_creative_install_ratio', 'app_first_cate_creative_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_app_first_second_cate_install_ratio_feat(start_date, end_date):
    print('extract first_second_cate install ratio features...')
    dump_path = './cache/app_first_second_cate_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = pd.merge(features, get_basic_APPcategories_feat(), on=['appID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['app_first_cate', 'app_second_cate', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['app_first_cate', 'app_second_cate'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['app_first_cate', 'app_second_cate'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'app_first_second_cate_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['app_first_cate', 'app_second_cate'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['app_first_second_cate_click_num'].replace(0, np.nan, inplace=True)
        features['app_first_second_cate_install_ratio'] = features['yes_num'] / features['app_first_second_cate_click_num']
        features['app_first_second_cate_install_ratio'].fillna(0, inplace=True)

        features = features[['app_first_cate', 'app_second_cate', 'app_first_second_cate_install_ratio', 'app_first_second_cate_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_app_second_cate_install_ratio_feat(start_date, end_date):
    print('extract second_cate install ratio features...')
    dump_path = './cache/app_second_cate_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = pd.merge(features, get_basic_APPcategories_feat(), on=['appID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['app_second_cate', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['app_second_cate'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['app_second_cate'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'app_second_cate_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['app_second_cate'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['app_second_cate_click_num'].replace(0, np.nan, inplace=True)
        features['app_second_cate_install_ratio'] = features['yes_num'] / features['app_second_cate_click_num']
        features['app_second_cate_install_ratio'].fillna(0, inplace=True)

        features = features[['app_second_cate', 'app_second_cate_install_ratio', 'app_second_cate_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_education_position_install_ratio_feat(start_date, end_date):
    print('extract education_position install ratio features...')
    dump_path = './cache/education_position_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['education', 'positionID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['education', 'positionID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['education', 'positionID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'education_position_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['education', 'positionID'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['education_position_click_num'].replace(0, np.nan, inplace=True)
        features['education_position_install_ratio'] = features['yes_num'] / features['education_position_click_num']
        features['education_position_install_ratio'].fillna(0, inplace=True)

        features = features[['education', 'positionID', 'education_position_install_ratio', 'education_position_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_user_position_install_ratio_feat(start_date, end_date):
    print('extract user_position install ratio features...')
    dump_path = './cache/user_position_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['userID', 'positionID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['userID', 'positionID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'user_position_yes_num'}, inplace=True)

        features_click_num = features.groupby(['userID', 'positionID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'user_position_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['userID', 'positionID'], how='left')

        features['user_position_yes_num'].fillna(0, inplace=True)
        features['user_position_click_num'].replace(0, np.nan, inplace=True)
        features['user_position_install_ratio'] = features['user_position_yes_num'] / features['user_position_click_num']
        features['user_position_install_ratio'].fillna(0, inplace=True)

        features = features[['userID', 'positionID', 'user_position_yes_num', 'user_position_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_user_telecomsOperator_install_ratio_feat(start_date, end_date):
    print('extract user_telecomsOperator install ratio features...')
    dump_path = './cache/user_telecomsOperator_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['userID', 'telecomsOperator', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['userID', 'telecomsOperator'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'user_telecomsOperator_yes_num'}, inplace=True)

        features_click_num = features.groupby(['userID', 'telecomsOperator'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'user_telecomsOperator_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['userID', 'telecomsOperator'], how='left')

        features['user_telecomsOperator_yes_num'].fillna(0, inplace=True)
        features['user_telecomsOperator_click_num'].replace(0, np.nan, inplace=True)
        features['user_telecomsOperator_install_ratio'] = features['user_telecomsOperator_yes_num'] / features['user_telecomsOperator_click_num']
        features['user_telecomsOperator_install_ratio'].fillna(0, inplace=True)

        features = features[['userID', 'telecomsOperator', 'user_telecomsOperator_yes_num',
                            'user_telecomsOperator_click_num', 'user_telecomsOperator_install_ratio']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_user_hour_install_ratio_feat(start_date, end_date):
    print('extract user_hour install ratio features...')
    dump_path = './cache/user_hour_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['userID', 'clickTime', 'label']]
        features['clickHour'] = features['clickTime'].map(lambda x : x // 10000 % 100)
        del features['clickTime']

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['userID', 'clickHour'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'user_clickHour_yes_num'}, inplace=True)

        features_click_num = features.groupby(['userID', 'clickHour'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'user_clickHour_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['userID', 'clickHour'], how='left')

        features['user_clickHour_yes_num'].fillna(0, inplace=True)
        features['user_clickHour_click_num'].replace(0, np.nan, inplace=True)
        features['user_clickHour_install_ratio'] = features['user_clickHour_yes_num'] / features['user_clickHour_click_num']
        features['user_clickHour_install_ratio'].fillna(0, inplace=True)

        features = features[['userID', 'clickHour', 'user_clickHour_yes_num', 'user_clickHour_click_num', 
                            'user_clickHour_install_ratio']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_user_app_first_cate_install_ratio_feat(start_date, end_date):
    print('extract user_app_first_cate install ratio features...')
    dump_path = './cache/user_app_first_cate_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = pd.merge(features, get_basic_APPcategories_feat(), on=['appID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['userID', 'app_first_cate', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['userID', 'app_first_cate'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'user_app_first_cate_yes_num'}, inplace=True)

        features_click_num = features.groupby(['userID', 'app_first_cate'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'user_app_first_cate_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['userID', 'app_first_cate'], how='left')

        features['user_app_first_cate_yes_num'].fillna(0, inplace=True)
        features['user_app_first_cate_click_num'].replace(0, np.nan, inplace=True)
        features['user_app_first_cate_install_ratio'] = features['user_app_first_cate_yes_num'] / features['user_app_first_cate_click_num']
        features['user_app_first_cate_install_ratio'].fillna(0, inplace=True)

        features = features[['userID', 'app_first_cate', 'user_app_first_cate_yes_num', 'user_app_first_cate_click_num', 
                            'user_app_first_cate_install_ratio']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_user_app_cate_install_ratio_feat(start_date, end_date):
    print('extract user_app_cate install ratio features...')
    dump_path = './cache/user_app_cate_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = pd.merge(features, get_basic_APPcategories_feat(), on=['appID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['userID', 'app_first_cate', 'app_second_cate', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['userID', 'app_first_cate', 'app_second_cate'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'user_app_cate_yes_num'}, inplace=True)

        features_click_num = features.groupby(['userID', 'app_first_cate', 'app_second_cate'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'user_app_cate_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['userID', 'app_first_cate', 'app_second_cate'], how='left')

        features['user_app_cate_yes_num'].fillna(0, inplace=True)
        features['user_app_cate_click_num'].replace(0, np.nan, inplace=True)
        features['user_app_cate_install_ratio'] = features['user_app_cate_yes_num'] / features['user_app_cate_click_num']
        features['user_app_cate_install_ratio'].fillna(0, inplace=True)

        features = features[['userID', 'app_first_cate', 'app_second_cate', 'user_app_cate_yes_num', 'user_app_cate_click_num', 
                            'user_app_cate_install_ratio']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_user_advertise_install_ratio_feat(start_date, end_date):
    print('extract user_advertiserID install ratio features...')
    dump_path = './cache/user_advertiserID_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['userID', 'advertiserID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['userID', 'advertiserID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'user_advertiserID_yes_num'}, inplace=True)

        features_click_num = features.groupby(['userID', 'advertiserID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'user_advertiserID_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['userID', 'advertiserID'], how='left')

        features['user_advertiserID_yes_num'].fillna(0, inplace=True)
        features['user_advertiserID_click_num'].replace(0, np.nan, inplace=True)
        features['user_advertiserID_install_ratio'] = features['user_advertiserID_yes_num'] / features['user_advertiserID_click_num']
        features['user_advertiserID_install_ratio'].fillna(0, inplace=True)

        features = features[['userID', 'advertiserID', 'user_advertiserID_yes_num', 'user_advertiserID_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_gender_haveBaby_install_ratio_feat(start_date, end_date):
    print('extract gender_haveBaby install ratio features...')
    dump_path = './cache/gender_haveBaby_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['gender', 'haveBaby', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['gender', 'haveBaby'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['gender', 'haveBaby'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'gender_haveBaby_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['gender', 'haveBaby'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['gender_haveBaby_click_num'].replace(0, np.nan, inplace=True)
        features['gender_haveBaby_install_ratio'] = features['yes_num'] / features['gender_haveBaby_click_num']
        features['gender_haveBaby_install_ratio'].fillna(0, inplace=True)

        features = features[['gender', 'haveBaby', 'gender_haveBaby_install_ratio', 'gender_haveBaby_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_gender_haveBaby_creative_install_ratio_feat(start_date, end_date):
    print('extract gender_haveBaby_creative install ratio features...')
    dump_path = './cache/gender_haveBaby_creative_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['gender', 'haveBaby', 'creativeID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['gender', 'haveBaby', 'creativeID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['gender', 'haveBaby', 'creativeID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'gender_haveBaby_creative_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['gender', 'haveBaby', 'creativeID'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['gender_haveBaby_creative_click_num'].replace(0, np.nan, inplace=True)
        features['gender_haveBaby_creative_install_ratio'] = features['yes_num'] / features['gender_haveBaby_creative_click_num']
        features['gender_haveBaby_creative_install_ratio'].fillna(0, inplace=True)

        features = features[['gender', 'haveBaby', 'creativeID', 'gender_haveBaby_creative_install_ratio', 'gender_haveBaby_creative_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_siteset_position_install_ratio_feat(start_date, end_date):
    print('extract siteset position install ratio features...')
    dump_path = './cache/siteset_position_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_position_feat(), on=['positionID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['sitesetID', 'positionID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['sitesetID', 'positionID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'siteset_position_yes_num'}, inplace=True)

        features_click_num = features.groupby(['sitesetID', 'positionID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'siteset_position_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['sitesetID', 'positionID'], how='left')

        features['siteset_position_yes_num'].fillna(0, inplace=True)
        features['siteset_position_click_num'].replace(0, np.nan, inplace=True)
        features['siteset_position_install_ratio'] = features['siteset_position_yes_num'] / features['siteset_position_click_num']
        features['siteset_position_install_ratio'].fillna(0, inplace=True)

        features = features[['sitesetID', 'positionID', 'siteset_position_install_ratio',
                            'siteset_position_click_num', 'siteset_position_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_siteset_positionType_install_ratio_feat(start_date, end_date):
    print('extract siteset positionType install ratio features...')
    dump_path = './cache/siteset_positionType_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_position_feat(), on=['positionID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['sitesetID', 'positionType', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['sitesetID', 'positionType'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'siteset_positionType_yes_num'}, inplace=True)

        features_click_num = features.groupby(['sitesetID', 'positionType'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'siteset_positionType_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['sitesetID', 'positionType'], how='left')

        features['siteset_positionType_yes_num'].fillna(0, inplace=True)
        features['siteset_positionType_click_num'].replace(0, np.nan, inplace=True)
        features['siteset_positionType_install_ratio'] = features['siteset_positionType_yes_num'] / features['siteset_positionType_click_num']
        features['siteset_positionType_install_ratio'].fillna(0, inplace=True)

        features = features[['sitesetID', 'positionType', 'siteset_positionType_install_ratio',
                            'siteset_positionType_click_num', 'siteset_positionType_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_positionType_position_install_ratio_feat(start_date, end_date):
    print('extract siteset position install ratio features...')
    dump_path = './cache/siteset_position_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_position_feat(), on=['positionID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['positionType', 'positionID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['positionType', 'positionID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'yes_num'}, inplace=True)

        features_click_num = features.groupby(['positionType', 'positionID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'type_position_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['positionType', 'positionID'], how='left')

        features['yes_num'].fillna(0, inplace=True)
        features['type_position_click_num'].replace(0, np.nan, inplace=True)
        features['type_position_install_ratio'] = features['yes_num'] / features['type_position_click_num']
        features['type_position_install_ratio'].fillna(0, inplace=True)

        features = features[['positionType', 'positionID', 'type_position_install_ratio', 'type_position_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_connectionType_position_install_ratio_feat(start_date, end_date):
    print('extract connectionType position install ratio features...')
    dump_path = './cache/connectionType_position_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['connectionType', 'positionID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['connectionType', 'positionID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'connectionType_position_yes_num'}, inplace=True)

        features_click_num = features.groupby(['connectionType', 'positionID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'connectionType_position_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['connectionType', 'positionID'], how='left')

        features['connectionType_position_yes_num'].fillna(0, inplace=True)
        features['connectionType_position_click_num'].replace(0, np.nan, inplace=True)
        features['connectionType_position_install_ratio'] = features['connectionType_position_yes_num'] / features['connectionType_position_click_num']
        features['connectionType_position_install_ratio'].fillna(0, inplace=True)

        features = features[['connectionType', 'positionID', 'connectionType_position_install_ratio',
                            'connectionType_position_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_creative_install_num(start_date, end_date):
    print('extract creative install num features...')
    dump_path = './cache/creative_install_num_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['creativeID', 'label']]

        features = features[features['label'] == 1]
        features = features.groupby(['creativeID'], as_index=False).count()
        features.rename(columns={'label' : 'creativeID_install_num'}, inplace=True)

        features = features[['creativeID', 'creativeID_install_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

# the number of apps installed
def get_app_install_number_feat(start_date, end_date):
    print('extract app install number features...')
    dump_path = './cache/app_install_number_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = get_user_app_info_feat(start_date, end_date)
        features = features.groupby(['appID'], as_index=False).count()
        features.rename(columns={'userID' : 'app_install_num'}, inplace=True)
        features.fillna(0, inplace=True)
        features.to_csv(dump_path,index=False, index_label=False)
    return features

# the install ratio of the user
def get_user_click_install_ration_feat(start_date, end_date):
    print('extract user install ratio features...')
    dump_path = './cache/user_click_install_ration_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = data_simulation(features, end_date)
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = features[['userID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['userID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'user_yes_num'}, inplace=True)

        features_click_num = features.groupby(['userID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'user_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['userID'], how='left')

        features['user_yes_num'].fillna(0, inplace=True)
        features['user_click_num'].replace(0, np.nan, inplace=True)
        features['user_install_ratio'] = features['user_yes_num'] / features['user_click_num']
        features['user_install_ratio'].fillna(0, inplace=True)

        features = features[['userID', 'user_install_ratio', 'user_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

# the number of apps the user installed
def get_user_install_app_number_feat(start_date, end_date):
    print('extract user install app number features...')
    dump_path = './cache/user_install_app_number_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = get_user_app_info_feat(start_date, end_date)
        features = features.groupby(['userID'], as_index=False).count()
        features.rename(columns={'appID' : 'user_install_num'}, inplace=True)
        features.fillna(0, inplace=True)
        features.to_csv(dump_path,index=False, index_label=False)
    return features

# the user preference of the app categaries
def get_user_preference_of_app_cate(start_date, end_date):
    print('extract user preference of the APP categories...')
    dump_path = './cache/user_preference_of_app_cate_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): user_cates = pd.read_csv(dump_path)
    else:
        app_cates = get_basic_APPcategories_feat()

        users_install_apps = get_user_app_info_feat(start_date, end_date)
        user_cates = pd.merge(users_install_apps, app_cates, on=['appID'], how='left')
        del app_cates, users_install_apps
        user_cates_1 = user_cates.groupby(['userID', 'app_first_cate'], as_index=False).count()
        user_cates_1.rename(columns={'appID':'first_cate_preference'}, inplace=True)
        user_cates_2 = user_cates.groupby(['userID', 'app_first_cate', 'app_second_cate'], as_index=False).count()
        user_cates_2.rename(columns={'appID':'cate_preference'}, inplace=True)

        user_cates = user_cates.groupby(['userID'], as_index=False).count()
        user_cates.rename(columns={'appID':'all_num'}, inplace=True)

        user_cates_1 = pd.merge(user_cates, user_cates_1[['userID', 'app_first_cate', 'first_cate_preference']], 
                                on=['userID', 'app_first_cate'], how='right')
        user_cates_1['first_cate_preference'].fillna(0, inplace=True)
        user_cates_1['all_num'].replace(0, np.nan, inplace=True)
        user_cates_1['first_cate_preference'] = user_cates_1['first_cate_preference'] / user_cates_1['all_num']
        del user_cates_1['app_second_cate']

        user_cates_2 = pd.merge(user_cates, user_cates_2[['userID', 'app_first_cate', 'app_second_cate', 'cate_preference']], 
                                on=['userID', 'app_first_cate', 'app_second_cate'], how='right')
        user_cates_2['cate_preference'].fillna(0, inplace=True)
        user_cates_2['all_num'].replace(0, np.nan, inplace=True)
        user_cates_2['cate_preference'] = user_cates_2['cate_preference'] / user_cates_2['all_num']

        user_cates = pd.merge(user_cates_2, user_cates_1, on=['userID', 'app_first_cate'], how='left')

        user_cates = user_cates[['userID', 'app_first_cate', 'app_second_cate', 'first_cate_preference', 'cate_preference']]
        user_cates.fillna(0, inplace=True)
        user_cates.to_csv(dump_path,index=False, index_label=False)
    return user_cates

# whether the user has installed the creative
def get_user_has_installed_app_feat(start_date, end_date):
    print('extract user has installed the app features...')
    dump_path = './cache/user_has_installed_app_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = get_user_app_info_feat(start_date, end_date)
        features.drop_duplicates(['userID', 'appID'], inplace=True)
        features['has_installed'] = 1
        features.to_csv(dump_path,index=False, index_label=False)
    return features

# the distance of the user has installed the app
def get_user_has_installed_distance_feat(start_date, end_date):
    print('extract the distance of the user has installed the app features...')
    dump_path = './cache/user_has_installed_distance_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:
        app_actions = pd.read_csv(user_app_actions_path)
        app_actions = app_actions[(app_actions['installTime'] >= 1000000) & (app_actions['installTime'] < 17000000)]
        app_actions = app_actions[(app_actions['installTime'] >= start_date) & (app_actions['installTime'] < end_date)]
        app_actions = app_actions[['userID', 'appID', 'installTime']]


        train = pd.read_csv(train_path)
        train = data_simulation(train, end_date)
        train = train[train['label'] == 1]
        train = train[(train['clickTime'] >= start_date) & (train['clickTime'] < end_date)]
        train = train[['userID', 'creativeID', 'clickTime']]
        creative = get_basic_ADcreative_feat()
        creative = creative[['creativeID', 'appID']]
        train = pd.merge(train, creative, on=['creativeID'], how='left')
        del creative
        train = train[['userID', 'appID', 'clickTime']]
        train.rename(columns={'clickTime' : 'installTime'}, inplace=True)

        features = pd.concat([app_actions, train])
        
        features = features.sort_values(by=['installTime'])
        features.drop_duplicates(['userID', 'appID'], inplace=True, keep='last')
        features['installTime'] = time2hours(end_date) - features['installTime'].map(time2hours)
        features.to_csv(dump_path,index=False, index_label=False)
    return features

# the number of the user has clicked
def get_the_num_of_user_has_clicked_feat(start_date, end_date, sub):
    print('extract the num of the user has clicked features from %d to %d...' % (start_date, end_date))
    dump_path = './cache/the_num_of_user_has_clicked_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        if sub:
            features = get_test_actions(start_date, end_date)
        else:
            features = get_actions(start_date, end_date)
        # features = data_simulation(features, end_date)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[['userID', 'appID', 'clickTime']]
        features = features.groupby(['userID', 'appID'], as_index=False).count()
        features.rename(columns={'clickTime' : 'click_app_num'}, inplace=True)
        features = features[['userID', 'appID', 'click_app_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_app_click_num_feat(start_date, end_date, sub):
    print('extract app click number features...')
    dump_path = './cache/app_click_number_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        if sub:
            features = get_test_actions(start_date, end_date)
        else:
            features = get_actions(start_date, end_date)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[['appID', 'userID']]
        features = features.groupby(['appID'], as_index=False).count()
        features.rename(columns={'userID' : 'the_current_day_app_click_num'}, inplace=True)
        features.fillna(0, inplace=True)
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_creative_click_num_feat(start_date, end_date, sub):
    print('extract creativeID click number features...')
    dump_path = './cache/creative_click_number_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        if sub:
            features = get_test_actions(start_date, end_date)
        else:
            features = get_actions(start_date, end_date)
        features = features[['creativeID', 'userID']]
        features = features.groupby(['creativeID'], as_index=False).count()
        features.rename(columns={'userID' : 'the_current_day_creativeID_click_num'}, inplace=True)
        features.fillna(0, inplace=True)
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_user_app_click_types_feat(start_date, end_date, sub):
    print('extract user_app_click_types features...')
    dump_path = './cache/user_app_click_types_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        if sub:
            features = get_test_actions(start_date, end_date)
        else:
            features = get_actions(start_date, end_date)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[['appID', 'userID']]
        features = features.drop_duplicates(['userID', 'appID'])
        features = features.groupby(['userID'], as_index=False).count()
        features.rename(columns={'appID' : 'user_app_click_types'}, inplace=True)
        features.fillna(0, inplace=True)
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_the_num_of_user_has_clicked_creative_feat(start_date, end_date, sub):
    print('extract the num of the user has clicked creative features from %d to %d...' % (start_date, end_date))
    dump_path = './cache/the_num_of_user_has_clicked_creative_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        if sub:
            features = get_test_actions(start_date, end_date)
        else:
            features = get_actions(start_date, end_date)
        # features = data_simulation(features, end_date)
        # features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[['userID', 'creativeID', 'clickTime']]
        features = features.groupby(['userID', 'creativeID'], as_index=False).count()
        features.rename(columns={'clickTime' : 'click_creativeID_num'}, inplace=True)
        features = features[['userID', 'creativeID', 'click_creativeID_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_user_click_num_feat(start_date, end_date):
    print('extract user click num features...')
    dump_path = './cache/user_click_num_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = get_actions(start_date, end_date)
        features = data_simulation(features, end_date)
        features = features[['userID', 'clickTime']]
        features = features.groupby(['userID'], as_index=False).count()
        features.rename(columns={'clickTime' : 'user_click_num'}, inplace=True)
        features = features[['userID','user_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

# the number of the user has clicked
def get_rank_the_num_of_user_has_clicked_feat(actions):
    print('extract the num of the user has clicked in the before day features...')
    dump_path = './cache/rank_the_num_of_user_has_clicked_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = get_actions(start_date, end_date)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[['userID', 'appID', 'clickTime']]
        rank = features.groupby(['userID','appID']).rank()
        rank.rename(columns={'clickTime' : 'click_rank'}, inplace=True)
        features = pd.concat([features[['userID', 'appID']], rank], axis=1)
        features = features[['userID', 'appID', 'click_rank']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

# user_install_the_same_app_num
def get_user_istall_the_same_app_num_feat(start_date, end_date):
    print('extract user install the same app number features...')
    dump_path = './cache/user_istall_the_same_app_num_feat_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = get_user_app_info_feat(start_date, end_date)
        features['the_same_app_install_num'] = 0
        features = features.groupby(['userID', 'appID'], as_index=False).count()
        features.fillna(0, inplace=True)
        features.to_csv(dump_path,index=False, index_label=False)
    return features

# the twice install ratio of the app
def get_app_twice_install_ratio_feat(start_date, end_date):
    print('extract app twice install ratio features...')
    dump_path = './cache/app_twice_install_ratio_feat.csv'
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = get_user_app_info_feat(start_date, end_date)
        features['num'] = 0
        # features = features.groupby(['appID', 'userID'], as_index=False).count()
        if True:   # the features is too big to map the entire features
            l = int(len(features) / 5)
            features_1 = features[0 : l]
            features_2 = features[l : l * 2]
            features_3 = features[l * 2 : l * 3]
            features_4 = features[l * 3 : l * 4]
            features_5 = features[l * 4 : ]
            del features
            print('features_1 groupby...')
            features_1 = features_1.groupby(['appID', 'userID'], as_index=False).count()
            features_1.rename(columns={'num' : 'num_1'}, inplace=True)
            print('features_2 groupby...')
            features_2 = features_2.groupby(['appID', 'userID'], as_index=False).count()
            features_2.rename(columns={'num' : 'num_2'}, inplace=True)
            features = pd.merge(features_1, features_2, on=['appID', 'userID'], how='outer')
            del features_1, features_2
            features.fillna(0, inplace=True)
            features['num'] = features['num_1'] + features['num_2']
            del features['num_1'], features['num_2']
            print('features_3 groupby...')
            features_3 = features_3.groupby(['appID', 'userID'], as_index=False).count()
            features_3.rename(columns={'num' : 'num_3'}, inplace=True)
            features = pd.merge(features, features_3, on=['appID', 'userID'], how='outer')
            del features_3
            features.fillna(0, inplace=True)
            features['num'] = features['num'] + features['num_3']
            del features['num_3']
            print('features_4 groupby...')
            features_4 = features_4.groupby(['appID', 'userID'], as_index=False).count()
            features_4.rename(columns={'num' : 'num_4'}, inplace=True)
            features = pd.merge(features, features_4, on=['appID', 'userID'], how='outer')
            del features_4
            features.fillna(0, inplace=True)
            features['num'] = features['num'] + features['num_4']
            del features['num_4']
            print('features_5 groupby...')
            features_5 = features_5.groupby(['appID', 'userID'], as_index=False).count()
            features_5.rename(columns={'num' : 'num_5'}, inplace=True)
            features = pd.merge(features, features_5, on=['appID', 'userID'], how='outer')
            del features_5
            features.fillna(0, inplace=True)
            features['num'] = features['num'] + features['num_5']
            del features['num_5']

        del features['userID']
        features_twice = features[features['num'] >= 2]
        features_twice = features_twice.groupby(['appID'], as_index=False).count()
        features_twice.rename(columns={'num' : 'twice_install_num'}, inplace=True)

        if True:
            l = int(len(features) / 4)
            features_1 = features[0 : l]
            features_2 = features[l : l * 2]
            features_3 = features[l * 2 : l * 3]
            features_4 = features[l * 3 : ]
            del features
            print('features_1 groupby...')
            features_1 = features_1.groupby(['appID'], as_index=False).count()
            features_1.rename(columns={'num' : 'install_num_1'}, inplace=True)
            print('features_2 groupby...')
            features_2 = features_2.groupby(['appID'], as_index=False).count()
            features_2.rename(columns={'num' : 'install_num_2'}, inplace=True)
            features = pd.merge(features_1, features_2, on=['appID'], how='outer')
            del features_1, features_2
            features.fillna(0, inplace=True)
            features['install_num'] = features['install_num_1'] + features['install_num_2']
            del features['install_num_1'], features['install_num_2']
            print('features_3 groupby...')
            features_3 = features_3.groupby(['appID'], as_index=False).count()
            features_3.rename(columns={'num' : 'install_num_3'}, inplace=True)
            features = pd.merge(features, features_3, on=['appID'], how='outer')
            del features_3
            features.fillna(0, inplace=True)
            features['install_num'] = features['install_num'] + features['install_num_3']
            del features['install_num_3']
            print('features_4 groupby...')
            features_4 = features_4.groupby(['appID'], as_index=False).count()
            features_4.rename(columns={'num' : 'install_num_4'}, inplace=True)
            features = pd.merge(features, features_4, on=['appID'], how='outer')
            del features_4
            features.fillna(0, inplace=True)
            features['install_num'] = features['install_num'] + features['install_num_4']
            del features['install_num_4']
            # features = features.groupby(['appID'], as_index=False).count()
            # features.rename(columns={'num' : 'install_num'}, inplace=True)

        features = pd.merge(features, features_twice, on=['appID'], how='left')

        features['twice_install_num'].fillna(0, inplace=True)
        features['install_num'].replace(0, np.nan, inplace=True)
        features['twice_install_ratio'] = features['twice_install_num'] / features['install_num']
        features['twice_install_ratio'].fillna(0, inplace=True)

        features = features[['appID', 'twice_install_ratio']]
        # features.to_csv(dump_path,index=False, index_label=False)
    return features

# user_app click rank
def get_user_app_click_rank(features, start_date, end_date):
    print('extract user app click rank features...')
    dump_path = './cache/user_app_click_rank_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:   
        rank = features[['userID', 'appID', 'clickTime']]
        rank = rank.groupby(['userID','appID']).rank()
        rank.rename(columns={'clickTime' : 'click_rank'}, inplace=True)
        features = pd.concat([features, rank], axis=1)
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_creative_active_hours_feat(start_date, end_date):
    print('extract creative active hours features...')
    dump_path = './cache/creative_active_hours_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:   
        features = pd.read_csv(train_path)
        features = features[['creativeID', 'clickTime']]
        features['clickTime'] = features['clickTime'].map(lambda x : x // 10000 % 100)
        # features['active_hours'] = 0
        features = features.drop_duplicates(['clickTime', 'creativeID'])
        features = features.groupby(['creativeID'], as_index=False).count()
        features.rename(columns={'clickTime' : 'creative_active_hours'}, inplace=True)
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_app_active_minutes_feat(start_date, end_date):
    print('extract app active minutes features...')
    dump_path = './cache/app_active_minutes_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:   
        features = pd.read_csv(train_path)
        features = features[(features['clickTime'] >= start_date) & (features['clickTime'] < end_date)]
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[['appID', 'clickTime']]
        features['clickTime'] = features['clickTime'].map(lambda x : x // 100 % 100)
        # features['active_hours'] = 0
        features = features.drop_duplicates(['clickTime', 'appID'])
        features = features.groupby(['appID'], as_index=False).count()
        features.rename(columns={'clickTime' : 'app_active_minutes'}, inplace=True)
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_app_active_hours_feat(start_date, end_date):
    print('extract app active hours features...')
    dump_path = './cache/app_active_hours_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:   
        features = pd.read_csv(train_path)
        features = features[(features['clickTime'] >= start_date) & (features['clickTime'] < end_date)]
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[['appID', 'clickTime']]
        features['clickTime'] = features['clickTime'].map(lambda x : x // 10000 % 100)
        # features['active_hours'] = 0
        features = features.drop_duplicates(['clickTime', 'appID'])
        features = features.groupby(['appID'], as_index=False).count()
        features.rename(columns={'clickTime' : 'app_active_hours'}, inplace=True)
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_app_active_distance_feat(start_date, end_date):
    print('extract app active distance features...')
    dump_path = './cache/app_active_distance_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:   
        features = pd.read_csv(train_path)
        features = features[(features['clickTime'] >= start_date) & (features['clickTime'] < end_date)]
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[['appID', 'clickTime']]
        features['clickTime'] = features['clickTime'].map(time2hours)
        features['active_hours'] = 0
        features = features.groupby(['clickTime', 'appID'], as_index=False).count()
        features = features.sort_values(['active_hours'])
        features.drop_duplicates(['appID'], inplace=True, keep='last')
        features['active_distance'] = time2hours(end_date) - features['clickTime']
        features = features[['appID', 'active_distance']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_creative_active_distance_feat(start_date, end_date):
    print('extract creativeive active distance features...')
    dump_path = './cache/creative_active_distance_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:   
        features = pd.read_csv(train_path)
        features = features[(features['clickTime'] >= start_date) & (features['clickTime'] < end_date)]
        fetures = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[['creativeID', 'clickTime']]
        features['clickTime'] = features['clickTime'].map(time2hours)
        features['active_hours'] = 0
        features = features.groupby(['clickTime', 'creativeID'], as_index=False).count()
        features = features.sort_values(['active_hours'])
        features.drop_duplicates(['creativeID'], inplace=True, keep='last')
        # features['clickTime'] = features['clickTime'].map(lambda x : x // 100 * 24 + x % 100)
        features['active_distance'] = time2hours(end_date) - features['clickTime']
        features = features[['creativeID', 'active_distance']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_creative_active_time_feat(start_date, end_date):
    print('extract creative active time features...')
    dump_path = './cache/creative_active_time_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:   
        features = pd.read_csv(train_path)
        features = features[features['label'] == 1]
        features = features[(features['clickTime'] >= start_date) & (features['clickTime'] < end_date)]
        features = features[['creativeID', 'clickTime']]
        features['clickHour'] = features['clickTime'].map(lambda x : x // 10000 % 100)
        features['install_num_the_special_hour'] = 0
        features = features.groupby(['clickHour', 'creativeID'], as_index=False).count()
        del features['clickTime']
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_app_active_time_feat(start_date, end_date):
    print('extract app active time features...')
    dump_path = './cache/app_active_time_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:   
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[features['label'] == 1]
        features = features[(features['clickTime'] >= start_date) & (features['clickTime'] < end_date)]
        features = features[['appID', 'clickTime']]
        features['clickHour'] = features['clickTime'].map(lambda x : x // 10000 % 100)
        features['install_num_the_special_hour'] = 0
        features = features.groupby(['clickHour', 'appID'], as_index=False).count()
        del features['clickTime']
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_gender_age_marriageStatus_app_install_ratio_feat(start_date, end_date):
    print('extract gender_age_marriageStatus_app install ratio features...')
    dump_path = './cache/gender_age_marriageStatus_app_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['gender', 'age', 'marriageStatus', 'appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['gender', 'age', 'marriageStatus', 'appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'gender_age_marriageStatus_app_yes_num'}, inplace=True)

        features_click_num = features.groupby(['gender', 'age', 'marriageStatus', 'appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'gender_age_marriageStatus_app_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['gender', 'age', 'marriageStatus', 'appID'], how='left')

        features['gender_age_marriageStatus_app_yes_num'].fillna(0, inplace=True)
        features['gender_age_marriageStatus_app_click_num'].replace(0, np.nan, inplace=True)
        features['gender_age_marriageStatus_app_install_ratio'] = features['gender_age_marriageStatus_app_yes_num'] / features['gender_age_marriageStatus_app_click_num']
        features['gender_age_marriageStatus_app_install_ratio'].fillna(0, inplace=True)

        features = features[['gender', 'age', 'marriageStatus', 'appID', 'gender_age_marriageStatus_app_install_ratio',
                             'gender_age_marriageStatus_app_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_residence_first_app_install_ratio_feat(start_date, end_date):
    print('extract residence_first_app install ratio features...')
    dump_path = './cache/residence_first_app_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['residence_first', 'appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['residence_first', 'appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'residence_first_app_yes_num'}, inplace=True)

        features_click_num = features.groupby(['residence_first', 'appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'residence_first_app_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['residence_first', 'appID'], how='left')

        features['residence_first_app_yes_num'].fillna(0, inplace=True)
        features['residence_first_app_click_num'].replace(0, np.nan, inplace=True)
        features['residence_first_app_install_ratio'] = features['residence_first_app_yes_num'] / features['residence_first_app_click_num']
        features['residence_first_app_install_ratio'].fillna(0, inplace=True)

        features = features[['residence_first', 'appID', 'residence_first_app_install_ratio',
                             'residence_first_app_click_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_user_app_install_num_feat(start_date, end_date):
    print('extract user_app_install_num features...')
    dump_path = './cache/user_app_install_num_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['userID', 'appID', 'label']]

        features = features[features['label'] == 1]
        features = features.groupby(['userID', 'appID'], as_index=False).count()
        features.rename(columns={'label' : 'user_app_install_num'}, inplace=True)

        features = features[['userID', 'appID', 'user_app_install_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_fist_last_interval_feat(start_date, end_date):
    print('extract get_fist_last_interval features...')
    dump_path = './cache/get_fist_last_interval_%s-%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)

        features = features[['userID', 'creativeID', 'clickTime']]
        feature_first = features.drop_duplicates(['userID', 'creativeID'], keep='first')
        feature_last = features.drop_duplicates(['userID', 'creativeID'], keep='last')
        features = pd.merge(feature_first, feature_last, on=['userID', 'creativeID'], how='left')
        features['global_click_creative_first_last_interval'] = features['clickTime_y'].map(time2seconds) - features['clickTime_x'].map(time2seconds)
        features['global_click_distance'] = time2seconds(end_date) - features['clickTime_x'].map(time2seconds)
        features = features[['userID', 'creativeID', 'global_click_creative_first_last_interval', 'global_click_distance']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def get_marriageStatus_app_install_ratio_feat(start_date, end_date):
    print('extract marriageStatus_app install ratio features...')
    dump_path = './cache/marriageStatus_app_install_ratio_feat_%s-%s.csv' % (start_date, end_date)

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = pd.read_csv(train_path)
        features = pd.merge(features, get_basic_user_feat(), on=['userID'], how='left')
        features = pd.merge(features, get_basic_ADcreative_feat(), on=['creativeID'], how='left')
        features = features[ (features['clickTime'] >= start_date) & (features['clickTime'] < end_date) ]
        features = data_simulation(features, end_date)
        features = features[['marriageStatus', 'appID', 'label']]

        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(['marriageStatus', 'appID'], as_index=False).count()
        features_click_yes.rename(columns={'label' : 'marriageStatus_app_yes_num'}, inplace=True)

        features_click_num = features.groupby(['marriageStatus', 'appID'], as_index=False).count()
        features_click_num.rename(columns={'label' : 'marriageStatus_app_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=['marriageStatus', 'appID'], how='left')

        features['marriageStatus_app_yes_num'].fillna(0, inplace=True)
        features['marriageStatus_app_click_num'].replace(0, np.nan, inplace=True)
        features['marriageStatus_app_install_ratio'] = features['marriageStatus_app_yes_num'] / features['marriageStatus_app_click_num']
        features['marriageStatus_app_install_ratio'].fillna(0, inplace=True)

        features = features[['marriageStatus', 'appID', 'marriageStatus_app_install_ratio',
                             'marriageStatus_app_click_num', 'marriageStatus_app_yes_num']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

def onehot(actions, cols):
    for col in cols:
        df = pd.get_dummies(actions[col], prefix=col)
        actions = pd.concat([actions, df], axis = 1)
        del actions[col]
    return actions

def make_train_set(start_date, end_date, test=False, sub=False):
    dump_path = './cache/train_set.csv'
    if os.path.exists(dump_path): actions = pd.read_csv(dump_path)
    else:
        actions_path = './cache/actions_%s-%s.csv' % (start_date, end_date)
        important_feat_path = './cache/important_feat_%s-%s.csv' % (start_date, end_date)

        history_start_date=00000000
        history_end_date = start_date
        yesterday = start_date - 1000000

        if os.path.exists(important_feat_path): actions = pd.read_csv(important_feat_path)
        else:
            if os.path.exists(actions_path): actions = pd.read_csv(actions_path)
            else:
                if sub:
                    actions = get_test_actions(start_date, end_date)
                else:
                    actions = get_actions(start_date, end_date)
                actions = pd.merge(actions, get_basic_user_feat(), how='left', on=['userID'])
                actions = pd.merge(actions, get_basic_ADcreative_feat(), how='left', on=['creativeID'])

                if True: #rank the userID_appID pairs by the clickTime
                    features = actions[['userID', 'appID', 'clickTime']]
                    features = features[features.duplicated(['userID', 'appID'], keep=False)]
                    rank = features.groupby(['userID','appID']).rank()
                    rank.rename(columns={'clickTime' : 'click_app_rank'}, inplace=True)
                    actions = pd.merge(actions, rank, left_index = True, right_index = True, how='outer')

                if True: #rank the userID_appID pairs by the clickTime
                    features = actions[['userID', 'creativeID', 'clickTime']]
                    features = features[features.duplicated(['userID', 'creativeID'], keep=False)]
                    rank = features.groupby(['userID','creativeID']).rank()
                    rank.rename(columns={'clickTime' : 'click_creativeID_rank'}, inplace=True)
                    actions = pd.merge(actions, rank, left_index = True, right_index = True, how='outer')
                    
                if True: #rank the userID_appID pairs by the clickTime
                    features = actions[['userID', 'creativeID', 'clickTime']]
                    duplicate = features.duplicated()
                    del features
                    duplicate.name = 'duplicates'
                    duplicate = duplicate.map(lambda x : int(x))
                    actions = pd.concat([actions, duplicate], axis=1)
                actions.to_csv(actions_path,index=False, index_label=False)

            actions = pd.merge(actions, get_basic_position_feat(), how='left', on=['positionID'])
            actions = pd.merge(actions, get_basic_APPcategories_feat(), how='left', on=['appID'])

            actions = pd.merge(actions, get_app_install_ratio_feat(history_start_date, history_end_date),
                                    on=['appID'], how='left')

            actions = pd.merge(actions, get_app_install_number_feat(history_start_date, history_end_date),
                                    on=['appID'], how='left')

            actions = pd.merge(actions, get_creative_install_num(history_start_date, history_end_date), 
                                    on=['creativeID'], how='left')

            actions = pd.merge(actions, get_user_install_app_number_feat(history_start_date, history_end_date),
                                    on=['userID'], how='left')

            actions = pd.merge(actions, get_creative_install_ratio_feat(history_start_date, history_end_date),
                                    on=['creativeID'], how='left')

            actions = pd.merge(actions, get_connectionType_position_install_ratio_feat(history_start_date, history_end_date), 
                                    on=['connectionType', 'positionID'], how='left')

            actions = pd.merge(actions, get_position_install_ratio_feat(history_start_date, history_end_date),
                                    on=['positionID'], how='left')

            actions = pd.merge(actions, get_user_position_install_ratio_feat(history_start_date, history_end_date),
                                on=['userID', 'positionID'], how='left')

            actions = pd.merge(actions, get_the_num_of_user_has_clicked_feat(start_date, end_date,sub),
                                on=['userID', 'appID'], how='left')

            actions = pd.merge(actions, get_user_click_install_ration_feat(history_start_date, history_end_date),
                                    on=['userID'], how='left', suffixes=('_app', '_user'))

            actions = pd.merge(actions, get_camgaign_install_ratio_feat(history_start_date, history_end_date),
                                    on=['camgaignID'], how='left')

            actions = pd.merge(actions, get_ad_install_ratio_feat(history_start_date, history_end_date), 
                                    on=['adID'], how='left')

            actions = pd.merge(actions, get_app_active_hours_feat(history_start_date, history_end_date), 
                                    on=['appID'], how='left')

            actions = pd.merge(actions, get_position_app_install_ratio_feat(history_start_date, history_end_date),
                                    on=['appID', 'positionID'], how='left')

            actions = pd.merge(actions, get_position_creative_install_ratio_feat(history_start_date, history_end_date),
                                    on=['positionID', 'creativeID'], how='left')

            actions = pd.merge(actions, get_gender_app_install_ratio_feat(history_start_date, history_end_date),
                                    on=['gender', 'appID'], how='left')

            actions = pd.merge(actions, get_user_has_installed_app_feat(history_start_date, history_end_date),
                                    on=['userID', 'appID'], how='left')

            actions = pd.merge(actions, get_user_has_installed_distance_feat(history_start_date, history_end_date),
                                    on=['userID', 'appID'], how='left')

            actions = pd.merge(actions, get_app_click_num_feat(start_date, end_date, sub),
                                    on=['appID'], how='left')

            # actions = pd.merge(actions, get_creative_click_num_feat(start_date, end_date, sub),
            #                         on=['creativeID'], how='left')

            features = actions[['userID', 'creativeID', 'clickTime']]
            feature_first = features.drop_duplicates(['userID', 'creativeID'], keep='first')
            feature_last = features.drop_duplicates(['userID', 'creativeID'], keep='last')
            features = pd.merge(feature_first, feature_last, on=['userID', 'creativeID'], how='left')
            features['click_creative_first_last_interval'] = features['clickTime_y'].map(time2seconds) - features['clickTime_x'].map(time2seconds)
            features = features[['userID', 'creativeID', 'click_creative_first_last_interval']]
            actions = pd.merge(actions, features, on=['userID', 'creativeID'], how='left')

            features = actions[['userID', 'appID', 'clickTime']]
            feature_first = features.drop_duplicates(['userID', 'appID'], keep='first')
            feature_last = features.drop_duplicates(['userID', 'appID'], keep='last')
            features = pd.merge(feature_first, feature_last, on=['userID', 'appID'], how='left')
            features['click_app_first_last_interval'] = features['clickTime_y'].map(time2seconds) - features['clickTime_x'].map(time2seconds)
            features = features[['userID', 'appID', 'click_app_first_last_interval']]
            actions = pd.merge(actions, features, on=['userID', 'appID'], how='left')

            features = actions[['userID', 'creativeID', 'clickTime']]
            duplicate = features.duplicated(['userID', 'creativeID'], keep=False)
            del features
            duplicate.name = 'all_duplicates'
            duplicate = duplicate.map(lambda x : int(x))
            actions = pd.concat([actions, duplicate], axis=1)

            features = actions[['userID', 'creativeID', 'clickTime']]
            duplicate = features.duplicated(['userID', 'creativeID'], keep='first')
            del features
            duplicate.name = 'first_duplicates'
            duplicate = duplicate.map(lambda x : int(x))
            actions = pd.concat([actions, duplicate], axis=1)

            features = actions[['userID', 'creativeID', 'clickTime']]
            duplicate = features.duplicated(['userID', 'creativeID'], keep='last')
            del features
            duplicate.name = 'last_duplicates'
            duplicate = duplicate.map(lambda x : int(x))
            actions = pd.concat([actions, duplicate], axis=1)

            actions['first_click'] = 0
            actions['last_click'] = 0
            actions['medium_click'] = 0
            actions.ix[(actions['all_duplicates']==1) & (actions['last_duplicates']==0), 'first_click'] = 1
            actions.ix[(actions['all_duplicates']==1) & (actions['first_duplicates']==0), 'last_click'] = 1
            actions.ix[(actions['all_duplicates']==1) & (actions['last_duplicates']==1) & (actions['last_duplicates']==1), 'medium_click'] = 1
            del actions['first_duplicates']
            del actions['last_duplicates']
            del actions['all_duplicates']

            actions.to_csv(important_feat_path,index=False, index_label=False)
        #============================================================================
        # features = actions[['userID', 'creativeID', 'clickTime']]
        # feature_first = features[features.duplicated(['userID', 'creativeID'], keep='first')]
        # feature_first.rename(columns={'clickTime' : 'current_click'}, inplace=True)
        # feature_last = features[features.duplicated(['userID', 'creativeID'], keep='last')]
        # feature_last.rename(columns={'clickTime' : 'last_click'}, inplace=True)
        # feature_first['last_click'] = feature_last['last_click'].values
        # feature_first['user_creative_last_interval'] = feature_first['current_click'].map(time2seconds) - feature_first['last_click'].map(time2seconds)
        # feature_first.rename(columns={'current_click' : 'clickTime'}, inplace=True)
        # del feature_first['last_click']
        # actions = pd.merge(actions, feature_first, on=['userID', 'creativeID', 'clickTime'], how='left')

        # features = actions[['userID', 'appID', 'clickTime']]
        # feature_first = features[features.duplicated(['userID', 'appID'], keep='first')]
        # feature_first.rename(columns={'clickTime' : 'current_click'}, inplace=True)
        # feature_last = features[features.duplicated(['userID', 'appID'], keep='last')]
        # feature_last.rename(columns={'clickTime' : 'last_click'}, inplace=True)
        # feature_first['last_click'] = feature_last['last_click'].values
        # feature_first['user_app_last_interval'] = feature_first['current_click'].map(time2seconds) - feature_first['last_click'].map(time2seconds)
        # feature_first.rename(columns={'current_click' : 'clickTime'}, inplace=True)
        # del feature_first['last_click']
        # actions = pd.merge(actions, feature_first, on=['userID', 'appID', 'clickTime'], how='left')

        # features = actions[['userID', 'clickTime']]
        # feature_first = features[features.duplicated(['userID'], keep='first')]
        # feature_first.rename(columns={'clickTime' : 'current_click'}, inplace=True)
        # feature_last = features[features.duplicated(['userID'], keep='last')]
        # feature_last.rename(columns={'clickTime' : 'last_click'}, inplace=True)
        # feature_first['last_click'] = feature_last['last_click'].values
        # feature_first['user_last_interval'] = feature_first['current_click'].map(time2seconds) - feature_first['last_click'].map(time2seconds)
        # feature_first.rename(columns={'current_click' : 'clickTime'}, inplace=True)
        # del feature_first['last_click']
        # actions = pd.merge(actions, feature_first, on=['userID', 'clickTime'], how='left')
        #============================================================================
        actions.fillna(0, inplace=True)


        del actions['positionID']
        del actions['camgaignID']

        del actions['userID']
        if not sub:
            del actions['conversionTime']
        del actions['clickTime']
        del actions['adID']
        del actions['appID']
        del actions['creativeID']

        labels = actions['label'].copy()
        del actions['label']

    return actions, labels

def make_train_set_LR_feat(start_date, end_date, test=False, sub=False):
    dump_path = './cache/train_set.csv'
    if os.path.exists(dump_path): actions = pd.read_csv(dump_path)
    else:
        actions_path = './cache/actions_%s-%s.csv' % (start_date, end_date)

        history_start_date=00000000
        history_end_date = start_date
        yesterday = start_date - 1000000

        if os.path.exists(actions_path): actions = pd.read_csv(actions_path)
        else:
            if sub:
                actions = get_test_actions(start_date, end_date)
            else:
                actions = get_actions(start_date, end_date)
            actions = pd.merge(actions, get_basic_user_feat(), how='left', on=['userID'])
            actions = pd.merge(actions, get_basic_ADcreative_feat(), how='left', on=['creativeID'])

            if True: #rank the userID_appID pairs by the clickTime
                features = actions[['userID', 'appID', 'clickTime']]
                rank = features.groupby(['userID','appID']).rank()
                rank.rename(columns={'clickTime' : 'click_app_rank'}, inplace=True)
                actions = pd.concat([actions, rank], axis=1)
                
            if True: #rank the userID_appID pairs by the clickTime
                features = actions[['userID', 'creativeID', 'clickTime']]
                rank = features.groupby(['userID','creativeID']).rank()
                rank.rename(columns={'clickTime' : 'click_creative_rank'}, inplace=True)
                actions = pd.concat([actions, rank], axis=1)
                
            if True: #rank the userID_appID pairs by the clickTime
                features = actions[['userID', 'creativeID', 'clickTime']]
                duplicate = features.duplicated()
                del features
                duplicate.name = 'duplicates'
                duplicate = duplicate.map(lambda x : int(x))
                actions = pd.concat([actions, duplicate], axis=1)
            actions.to_csv(actions_path,index=False, index_label=False)

        actions = pd.merge(actions, get_basic_position_feat(), how='left', on=['positionID'])
        actions = pd.merge(actions, get_basic_APPcategories_feat(), how='left', on=['appID'])

        actions = pd.merge(actions, get_app_install_ratio_feat(history_start_date, history_end_date),
                                on=['appID'], how='left')

        actions = pd.merge(actions, get_app_install_number_feat(history_start_date, history_end_date),
                                on=['appID'], how='left')

        actions = pd.merge(actions, get_creative_install_num(history_start_date, history_end_date), 
                                on=['creativeID'], how='left')

        actions = pd.merge(actions, get_user_install_app_number_feat(history_start_date, history_end_date),
                                on=['userID'], how='left')

        actions = pd.merge(actions, get_creative_install_ratio_feat(history_start_date, history_end_date),
                                on=['creativeID'], how='left')

        actions = pd.merge(actions, get_connectionType_position_install_ratio_feat(history_start_date, history_end_date), 
                                on=['connectionType', 'positionID'], how='left')

        actions = pd.merge(actions, get_position_install_ratio_feat(history_start_date, history_end_date),
                                on=['positionID'], how='left')

        actions = pd.merge(actions, get_user_position_install_ratio_feat(history_start_date, history_end_date),
                            on=['userID', 'positionID'], how='left')

        # actions = pd.merge(actions, get_user_preference_of_app_cate(history_start_date, history_end_date), 
        #                     on=['userID', 'app_first_cate', 'app_second_cate'], how='left')

        actions = pd.merge(actions, get_the_num_of_user_has_clicked_feat(start_date, end_date,sub),
                            on=['userID', 'appID'], how='left')

        actions = pd.merge(actions, get_user_click_install_ration_feat(history_start_date, history_end_date),
                                on=['userID'], how='left', suffixes=('_app', '_user'))

        actions = pd.merge(actions, get_camgaign_install_ratio_feat(history_start_date, history_end_date),
                                on=['camgaignID'], how='left')

        actions = pd.merge(actions, get_ad_install_ratio_feat(history_start_date, history_end_date), 
                                on=['adID'], how='left')

        actions = pd.merge(actions, get_app_active_hours_feat(history_start_date, history_end_date), 
                                on=['appID'], how='left')

        actions = pd.merge(actions, get_position_app_install_ratio_feat(history_start_date, history_end_date),
                                on=['appID', 'positionID'], how='left')

        # actions = pd.merge(actions, get_age_positionType_install_ration_feat(history_start_date, history_end_date),
        #                         on=['age', 'positionType'], how='left')
        # actions = pd.merge(actions, get_app_twice_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['appID'], how='left')
        # actions = pd.merge(actions, get_user_has_installed_app_feat(history_start_date, history_end_date),
        #                     on=['userID', 'appID'], how='left')
        # actions = pd.merge(actions, get_user_istall_the_same_app_num_feat(history_start_date, history_end_date),
        #                         on=['userID', 'appID'], how='left')
        # actions = pd.merge(actions, get_age_gender_appID_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['age', 'gender', 'appID'], how='left')

        # actions = pd.merge(actions, get_connectionType_position_app_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['connectionType', 'positionID', 'appID'], how='left')
        # actions = pd.merge(actions, get_age_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['age'], how='left')

        # actions = pd.merge(actions, get_age_camgaignID_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['age', 'camgaignID'], how='left')
        # actions = pd.merge(actions, get_user_telecomsOperator_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['userID', 'telecomsOperator'], how='left')

        # actions['clickHour'] = actions['clickTime'].map(lambda x : x // 10000 % 100)
        # actions = pd.merge(actions, get_user_hour_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['userID', 'clickHour'], how='left')
        # del actions['clickHour']

        # actions = pd.merge(actions, get_user_advertise_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['userID', 'advertiserID'], how='left')
        # actions = pd.merge(actions, get_user_has_installed_distance_feat(history_start_date, history_end_date),
        #                     on = ['userID', 'appID'], how='left')

        # actions = pd.merge(actions, get_age_app_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['age', 'appID'], how='left')
        # actions = pd.merge(actions, get_app_active_distance_feat(history_start_date, history_end_date),
        #                         on=['appID'], how='left')
        # actions = pd.merge(actions, get_creative_active_hours_feat(history_start_date, history_end_date),
        #                         on=['creativeID'], how='left')
        # actions = pd.merge(actions, get_app_active_time_feat(history_start_date, history_end_date),
        #                         on=['clickHour', 'appID'], how='left')

        # actions = pd.merge(actions, get_siteset_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['sitesetID'], how='left')
        # residences = get_basic_user_feat()
        # residences = residences[['userID', 'residence_first']]
        # actions = pd.merge(actions, residences, on=['userID'], how='left')
        # actions = pd.merge(actions, get_residence_first_app_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['residence_first', 'appID'], how='left')
        # del actions['residence_first']
        # actions = pd.merge(actions, get_gender_age_marriageStatus_app_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['gender', 'age', 'marriageStatus', 'appID'], how='left')
        # actions = pd.merge(actions, get_gender_app_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['gender', 'appID'], how='left')
        # actions = pd.merge(actions, get_age_cate_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['age', 'app_first_cate'], how='left')

        # actions = pd.merge(actions, get_haveBaby_appID_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['haveBaby', 'appID'], how='left')

        # actions = pd.merge(actions, get_education_connectionType_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['education', 'connectionType'], how='left')

        # actions = pd.merge(actions, get_age_connectionType_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['age', 'connectionType'], how='left')
        # actions = pd.merge(actions, get_age_creative_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['age', 'creativeID'], how='left')
        # actions = pd.merge(actions, get_advertise_install_ratio_feat(history_start_date, history_end_date), 
        #                         on=['advertiserID'], how='left')
        # actions = pd.merge(actions, get_creative_install_num(history_start_date, history_end_date), 
        #                         on=['creativeID'], how='left')
        # actions = pd.merge(actions, get_creative_active_distance_feat(history_start_date, history_end_date),
        #                         on=['creativeID'], how='left')
        # actions = pd.merge(actions, get_app_active_distance_feat(history_start_date, history_end_date),
        #                         on=['appID'], how='left')
        # actions = pd.merge(actions, get_app_active_minutes_feat(history_start_date, history_end_date), 
        #                     on=['appID'], how='left')

        # actions = pd.merge(actions, get_gender_creativeID_install_ratio_feat(history_start_date, history_end_date),
        #                         on=['gender', 'creativeID'], how='left')

        # actions = pd.merge(actions, get_app_install_number_feat(history_start_date, history_end_date),
        #                         on=['appID'], how='left')


        actions.fillna(0, inplace=True)
        del actions['userID']

        if not sub:
            del actions['conversionTime']
        del actions['clickTime']
        # del actions['positionID']
        # del actions['app_second_cate']
        # del actions['app_first_cate']
        # del actions['cate_preference']
        # del actions['camgaignID']
        del actions['adID']
        # del actions['advertiserID']
        del actions['appID']
        del actions['creativeID']
        # del actions['click_app_rank']
        # del actions['duplicates']

        labels = actions['label'].copy()
        del actions['label']

    return actions, labels

def get_install_feat(actions_df, train_step):
    if train_step:  dump_path = './cache/train_install.csv'
    else:           dump_path = './cache/test_install.csv'
    if os.path.exists(dump_path): actions_df = pd.read_csv(dump_path)
    else:
        installed_df = pd.read_csv(user_installedapps_path)
        installed_df['installed'] = 1.0
        actions_df = pd.merge(actions_df,installed_df,how='left',on=['userID','appID']); installed_df = []
        appAction_df = pd.read_csv(user_app_actions_path)
        appAction_df =  appAction_df.sort_values('installTime',ascending=True).reset_index(drop=True)
        appAction_df = appAction_df.groupby(['userID','appID'],as_index=False).first().reset_index(drop=True)
        actions_df = pd.merge(actions_df,appAction_df,how='left',on=['userID','appID']); appAction_df = []
        actions_df['installed'] = actions_df['installed'].replace(np.nan, 0.0)
        actions_df['installed'] = (actions_df['installed']==1.0) | (actions_df['clickTime'] >= actions_df['installTime'] )
        fun = lambda x: 1.0 if x else 0.0;
        actions_df['installed'] = actions_df['installed'].map(fun)
        actions_df = actions_df[['userID','appID','installed']]
        actions_df.to_csv(dump_path,index=False, index_label=False)
    return actions_df

def report( right_list, pre_list ):
    epsilon = 1e-15
    act = right_list
    pred = np.maximum(epsilon, pre_list)
    pred = np.minimum(1-epsilon, pre_list)
    ll = sum(act*np.log(pred) + np.subtract(1,act)*np.log(np.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return 'my_loss',ll

def featureCombine():
    '''
    basic : basic features
    k : the number of features to combine
    '''
    start_date = 24000000
    end_date = 25000000
    data = get_actions(start_date, end_date)
    data = pd.merge(data, get_basic_user_feat(), how='left', on=['userID'])
    data = pd.merge(data, get_basic_ADcreative_feat(), how='left', on=['creativeID'])
    data = pd.merge(data, get_basic_position_feat(), how='left', on=['positionID'])
    data = pd.merge(data, get_basic_APPcategories_feat(), how='left', on=['appID'])
    del data['clickTime']
    del data['conversionTime']


    labels = data['label']
    # del data['label']
    columns = list(data.columns)
    columns.remove('label')
    length = len(columns)

    for i in range(0, length, 1):
        for j in range(i+1, length, 1):
            for k in range(j+1, length, 1):
                print('i = %d\tj=%d\tk=%d' % (i, j, k))
                features = combine(data, [columns[i], columns[j], columns[k]])
                data = pd.merge(data, features, on=[columns[i], columns[j], columns[k]], how='left')

    data = data.ix[:, ~data.columns.isin(columns)]
    del data['label']
    data.fillna(0, inplace=True)
    return data, labels

def featureCombine_2():
    '''
    basic : basic features
    k : the number of features to combine
    '''
    start_date = 24000000
    end_date = 25000000
    data = get_actions(start_date, end_date)
    data = pd.merge(data, get_basic_user_feat(), how='left', on=['userID'])
    data = pd.merge(data, get_basic_ADcreative_feat(), how='left', on=['creativeID'])
    data = pd.merge(data, get_basic_position_feat(), how='left', on=['positionID'])
    data = pd.merge(data, get_basic_APPcategories_feat(), how='left', on=['appID'])
    del data['clickTime']
    del data['conversionTime']


    labels = data['label']
    # del data['label']
    columns = list(data.columns)
    columns.remove('label')
    length = len(columns)

    for i in range(0, length, 1):
        for j in range(i+1, length, 1):
            print('i = %d\tj=%d' % (i, j))
            features = combine(data, [columns[i], columns[j]])
            data = pd.merge(data, features, on=[columns[i], columns[j]], how='left')

    data = data.ix[:, ~data.columns.isin(columns)]
    del data['label']
    data.fillna(0, inplace=True)
    return data, labels


def combine(data, columns):
    print('combine ' + str(columns) + '...')
    dump_path = './cache/' + 'combine ' + str(columns) + '_%s-%s.csv'

    if os.path.exists(dump_path): features = pd.read_csv(dump_path)
    else:      
        features = data[columns + ['label']]
        features_click_yes = features[features['label'] == 1]
        features_click_yes = features_click_yes.groupby(columns, as_index=False).count()
        features_click_yes.rename(columns={'label' : str(columns) + '_yes_num'}, inplace=True)

        features_click_num = features.groupby(columns, as_index=False).count()
        features_click_num.rename(columns={'label' : str(columns) + '_click_num'}, inplace=True)

        features = pd.merge(features_click_num, features_click_yes, on=columns, how='left')

        features[str(columns) + '_yes_num'].fillna(0, inplace=True)
        features[str(columns) + '_click_num'].replace(0, np.nan, inplace=True)
        features[str(columns) + '_install_ratio'] = features[str(columns) + '_yes_num'] / features[str(columns) + '_click_num']
        features[str(columns) + '_install_ratio'].fillna(0, inplace=True)

        features = features[columns + [str(columns) + '_yes_num', 
                            str(columns) + '_click_num', str(columns) + '_install_ratio']]
        features.to_csv(dump_path,index=False, index_label=False)
    return features

if __name__ == '__main__':
    make_train_set(24000000, 25000000)
    for i in range(18000000, 31000000, 1000000):
        make_train_set(i,i+1000000)




# TODO:
# ratio = num_user_installed / num_user_click
# 星期信息
# 距离待测天的天数
# positionID 占比
# age?
# education?
# appcate ...
# 最大活跃距离现在时间
# ａｐｐ　活跃时间段

# active
# 注意，这里所有的click都默认天数没有类似01之类的，都为两位数天数

# 加上yes_num

# age cate 
# education ?
# age ?
# hometown ?


#冷启动
#平滑
#条件概率特征
# age app position gender