#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np

action_1_path = "./data/JData_Action_201602.csv"
action_2_path = "./data/JData_Action_201603.csv"
action_3_path = "./data/JData_Action_201604.csv"
comment_path = "./data/JData_Comment.csv"
product_path = "./data/JData_Product.csv"
user_path = "./data/JData_User.csv"

comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07", "2016-03-14",
                "2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]

def get_date(x):
    return x.split(' ')[0]

def strptime(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

def deltatime2hours(x):
    return x.days * 24 + x.seconds / 3600

def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1

def get_basic_user_feat():
    dump_path = './cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        user['age'] = user['age'].map(convert_age)
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
        pickle.dump(user, open(dump_path, 'w'))
    return user


def get_basic_product_feat():
    dump_path = './cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path))
    else:
        product = pd.read_csv(product_path)
        attr1_df = pd.get_dummies(product["a1"], prefix="a1")
        attr2_df = pd.get_dummies(product["a2"], prefix="a2")
        attr3_df = pd.get_dummies(product["a3"], prefix="a3")
        product = pd.concat([product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
        pickle.dump(product, open(dump_path, 'w'))
    return product


def get_actions_1():
    action = pd.read_csv(action_1_path)
    return action

def get_actions_2():
    action2 = pd.read_csv(action_2_path)
    return action2

def get_actions_3():
    action3 = pd.read_csv(action_3_path)
    return action3


def get_actions(start_date, end_date):
    """

    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    dump_path = './cache/all_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        action_1 = get_actions_1()
        action_1 = action_1[(action_1.time >= start_date) & (action_1.time < end_date)]
        del action_1['brand'], action_1['model_id'], action_1['cate']
        action_2 = get_actions_2()
        action_2 = action_2[(action_2.time >= start_date) & (action_2.time < end_date)]
        del action_2['brand'], action_2['model_id'], action_2['cate']
        actions = pd.concat([action_1, action_2])
        del action_1, action_2
        action_3 = get_actions_3()
        action_3 = action_3[(action_3.time >= start_date) & (action_3.time < end_date)]
        del action_3['brand'], action_3['model_id'], action_3['cate']
        actions = pd.concat([actions, action_3]) # type: pd.DataFrame
        del action_3
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_action_feat(start_date, end_date):
    dump_path = './cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[['user_id', 'sku_id', 'type']]
        df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        del actions['type']
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_accumulate_action_feat(start_date, end_date):
    dump_path = './cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1) # type: pd.DataFrame
        #近期行为按时间衰减
        actions['weights'] = actions['time'].map(lambda x: datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        #actions['weights'] = time.strptime(end_date, '%Y-%m-%d') - actions['datetime']
        actions['weights'] = actions['weights'].map(lambda x: math.exp(-x.days))
        print actions.head(10)
        actions['action_1'] = actions['action_1'] * actions['weights']
        actions['action_2'] = actions['action_2'] * actions['weights']
        actions['action_3'] = actions['action_3'] * actions['weights']
        actions['action_4'] = actions['action_4'] * actions['weights']
        actions['action_5'] = actions['action_5'] * actions['weights']
        actions['action_6'] = actions['action_6'] * actions['weights']
        del actions['model_id']
        del actions['type']
        del actions['time']
        del actions['datetime']
        del actions['weights']
        actions = actions.groupby(['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_comments_product_feat(end_date):
    dump_path = './cache/comments_accumulate_to_%s.pkl' % end_date
    if os.path.exists(dump_path):
        comments = pickle.load(open(dump_path))
    else:
        comments = pd.read_csv(comment_path)
        comment_date_end = end_date
        comment_date_begin = comment_date[0]
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
        df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, df], axis=1) # type: pd.DataFrame
        #del comments['dt']
        #del comments['comment_num']
        comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]
        pickle.dump(comments, open(dump_path, 'w'))
    return comments


def get_accumulate_user_feat(start_date, end_date):
    feature = ['user_id', 'user_action_1_ratio', 'user_action_2_ratio', 'user_action_3_ratio',
               'user_action_5_ratio', 'user_action_6_ratio', 'action_1', 'action_2', 'action_3',
               'action_4', 'action_5', 'action_6']
    dump_path = './cache/user_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['user_id'], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_ratio'] = (actions['action_4'] + 1) / (actions['action_1'] + 1)
        actions['user_action_2_ratio'] = (actions['action_4'] + 1) / (actions['action_2'] + 1)
        actions['user_action_3_ratio'] = (actions['action_4'] + 1) / (actions['action_3'] + 1)
        actions['user_action_5_ratio'] = (actions['action_4'] + 1) / (actions['action_5'] + 1)
        actions['user_action_6_ratio'] = (actions['action_4'] + 1) / (actions['action_6'] + 1)
        actions = actions[feature]
        pickle.dump(actions, open(dump_path, 'w'))
    return actions


def get_accumulate_product_feat(start_date, end_date):
    feature = ['sku_id', 'product_action_1_ratio', 'product_action_2_ratio', 'product_action_3_ratio',
               'product_action_5_ratio', 'product_action_6_ratio', 'action_1', 'action_2', 'action_3',
               'action_4', 'action_5', 'action_6']
    dump_path = './cache/product_feat_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions[['sku_id', 'user_id']], df], axis=1)

        product_twice_buy = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        product_buy = product_twice_buy[product_twice_buy['action_4'] >= 1]
        product_buy = product_buy.groupby(['sku_id'], as_index=False).count()
        product_buy.rename(columns={'user_id':'num_of_buy_user'},inplace=True)
        product_buy = product_buy[['sku_id', 'num_of_buy_user']]
        product_twice_buy = product_twice_buy[product_twice_buy['action_4'] >= 2]
        product_twice_buy = product_twice_buy.groupby(['sku_id'], as_index=False).count()
        product_twice_buy.rename(columns={'user_id':'num_of_twice_buy_user'},inplace=True)
        product_twice_buy = product_twice_buy[['sku_id', 'num_of_twice_buy_user']]
        products = pd.merge(product_buy, product_twice_buy, on=['sku_id'], how='left')
        del product_buy, product_twice_buy
        products['twice_buy_rate'] = (products['num_of_twice_buy_user'] + 1) / (products['num_of_buy_user'] + 1)
        del products['num_of_twice_buy_user'], products['num_of_buy_user']

        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_action_1_ratio'] = (actions['action_4'] + 1) / (actions['action_1'] + 1)
        actions['product_action_2_ratio'] = (actions['action_4'] + 1) / (actions['action_2'] + 1)
        actions['product_action_3_ratio'] = (actions['action_4'] + 1) / (actions['action_3'] + 1)
        actions['product_action_5_ratio'] = (actions['action_4'] + 1) / (actions['action_5'] + 1)
        actions['product_action_6_ratio'] = (actions['action_4'] + 1) / (actions['action_6'] + 1)
        actions = actions[feature]
        actions = pd.merge(actions, products, on=['sku_id'], how='left')
        pickle.dump(actions, open(dump_path, 'w'))
    return actions

def get_user_product_touch_feat(start_date, end_date):
    dump_path = './cache/user_product_touch_feat_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        feats = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        # actions['time'] = actions['time'].map(strptime)
        # actions['time'] = datetime.strptime(end_date, '%Y-%m-%d') - actions['time']
        # i = int(len(actions) / 2)
        # actions_1 = actions[0:i]
        # actions_2 = actions[i:]
        # del actions
        # actions_1['time'] = actions_1['time'].map(deltatime2hours)
        # actions_2['time'] = actions_2['time'].map(deltatime2hours)
        # actions = pd.concat([actions_1, actions_2])
        # del actions_1, actions_2

        actions = actions.sort_values(by=['time'])
        actions_first_touch = actions.drop_duplicates(['user_id', 'sku_id'], keep='last')
        actions_first_touch['time'] = (datetime.strptime(end_date, '%Y-%m-%d') - actions_first_touch['time'].map(strptime)).map(deltatime2hours)
        # del actions
        actions_first_touch = actions_first_touch[['user_id', 'sku_id', 'time']]
        actions_first_touch.rename(columns={'time':'first_touch'}, inplace=True)
        
        # actions = get_actions(start_date, end_date)
        actions_last_touch = actions.drop_duplicates(['user_id', 'sku_id'], keep='first')
        actions_last_touch['time'] = (datetime.strptime(end_date, '%Y-%m-%d') - actions_last_touch['time'].map(strptime)).map(deltatime2hours)
        del actions
        actions_last_touch = actions_last_touch[['user_id', 'sku_id', 'time']]
        actions_last_touch.rename(columns={'time':'last_touch'}, inplace=True)

        feats = pd.merge(actions_first_touch, actions_last_touch, on=['user_id', 'sku_id'], how='left')
        del actions_first_touch,actions_last_touch
        feats.fillna(np.inf, inplace=True)
        pickle.dump(feats, open(dump_path, 'w'))
    return feats

def get_product_cart_feat(start_date, end_date):
    'in the cart, has_been_deleted, how long from add the cart'
    dump_path = './cache/product_cart_feat_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)

        actions_deleted = actions[actions['type']==3]
        del actions_deleted['time'], actions_deleted['type']
        actions_deleted['has_been_deleted'] = 1

        actions = actions[actions['type'].isin([2,3,4])]
        actions = actions.sort_values(by=['time']).drop_duplicates(['user_id', 'sku_id'], keep='last')
        actions = actions[actions['type']==2]
        actions['in_cart'] = 1
        actions['time'] = actions['time'].map(strptime)
        actions['time'] = datetime.strptime(end_date, '%Y-%m-%d') - actions['time']
        actions['time'] = actions['time'].map(deltatime2hours)
        actions.rename(columns = {'time':'hours_in_cart'}, inplace=True) 

        actions = pd.merge(actions, actions_deleted, on=['user_id', 'sku_id'], how='outer')
        actions['hours_in_cart'].fillna(np.inf, inplace=True)
        actions['has_been_deleted'].fillna(0, inplace=True)
        actions['in_cart'].fillna(0, inplace=True)
        del actions['type']
        pickle.dump(actions, open(dump_path, 'w'))
    return actions

#-------------------------------------------------------------
def get_user_cate_feat(start_date, end_date):
    dump_path = './cache/user_cate_feat_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        action_1 = get_actions_1()
        action_1 = action_1[(action_1.time >= start_date) & (action_1.time < end_date)]
        del action_1['brand'], action_1['model_id'], action_1['time']
        action_2 = get_actions_2()
        action_2 = action_2[(action_2.time >= start_date) & (action_2.time < end_date)]
        del action_2['brand'], action_2['model_id'], action_2['time']
        actions = pd.concat([action_1, action_2])
        del action_1, action_2
        action_3 = get_actions_3()
        action_3 = action_3[(action_3.time >= start_date) & (action_3.time < end_date)]
        del action_3['brand'], action_3['model_id'], action_3['time']
        actions = pd.concat([actions, action_3]) # type: pd.DataFrame
        del action_3

        all_actions_reference = actions[['user_id', 'sku_id']]
        all_actions_reference = all_actions_reference.groupby(['user_id'], as_index=False).count()
        all_actions_reference.rename(columns={'sku_id':'all_actions'}, inplace=True)

        cate_actions_reference = actions[['user_id', 'cate']]
        cate_actions_reference = cate_actions_reference[cate_actions_reference.cate==8]
        cate_actions_reference = cate_actions_reference.groupby(['user_id'], as_index=False).count()
        cate_actions_reference.rename(columns={'cate':'cate_actions'}, inplace=True)

        actions_reference = pd.merge(all_actions_reference, cate_actions_reference, on=['user_id'], how='left')
        del all_actions_reference, cate_actions_reference
        actions_reference['cate__actions_preferce_rate'] = (actions_reference['cate_actions'] + 1) / (actions_reference['all_actions'] + 1)
        del actions_reference['cate_actions'], actions_reference['all_actions']

        actions = actions[actions['type']==4]
        all_buy_reference = actions[['user_id', 'type']]
        all_buy_reference = all_buy_reference.groupby(['user_id'], as_index=False).count()
        all_buy_reference.rename(columns={'type':'all_buy_actions'}, inplace=True)

        cate_buy_reference = actions[['user_id', 'cate']]
        del actions
        cate_buy_reference = cate_buy_reference[cate_buy_reference.cate==8]
        cate_buy_reference = cate_buy_reference.groupby(['user_id'], as_index=False).count()
        cate_buy_reference.rename(columns={'cate':'cate_buy_actions'}, inplace=True)

        buy_reference = pd.merge(all_buy_reference, cate_buy_reference, on=['user_id'], how='left')
        del all_buy_reference, cate_buy_reference
        buy_reference['cate_buy_reference'] = (buy_reference['cate_buy_actions'] + 1) / (buy_reference['all_buy_actions'] + 1)
        del buy_reference['cate_buy_actions'], buy_reference['all_buy_actions']

        actions = pd.merge(actions_reference, buy_reference, on=['user_id'], how='left')
        pickle.dump(actions, open(dump_path, 'w'))
    return actions

def get_user_active_feat(start_date, end_date):
    dump_path = './cache/user_active_feat_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions['time'] = actions['time'].map(get_date)

        all_actions = actions[['user_id', 'time']]
        all_actions.drop_duplicates(['user_id', 'time'], keep='first', inplace=True)
        all_actions = all_actions.groupby(['user_id'], as_index=False).count()
        all_actions.rename(columns={'time':'active_days'}, inplace=True)

        actions = actions[actions['type']==4]
        buy_actions = actions[['user_id', 'time']]
        del actions
        buy_actions.drop_duplicates(['user_id', 'time'], keep='first', inplace=True)
        buy_actions = buy_actions.groupby(['user_id'], as_index=False).count()
        buy_actions.rename(columns={'time':'active_buy_days'}, inplace=True)

        actions = pd.merge(all_actions, buy_actions, on=['user_id'], how='left')
        del all_actions, buy_actions
        actions['active_buy_rate'] = (actions['active_buy_days'] + 1) / (actions['active_days'] + 1)

        pickle.dump(actions, open(dump_path, 'w'))
    return actions

def get_user_product_info_preference_feat(start_date, end_date):
    dump_path = './cache/user_product_info_preference_feat_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions.type==4]
        actions = actions[['user_id', 'sku_id']]
        products = get_basic_product_feat()
        del products['cate'], products['brand']
        actions = pd.merge(actions, products, on=['sku_id'], how='left')
        del actions['sku_id']
        actions = actions.groupby(['user_id'], as_index=False).sum()

        actions['a1_-1_preference'] = (actions['a1_-1'] + 1) / (actions['a1_-1'] + actions['a1_1'] + actions['a1_2'] + actions['a1_3'] + 1)
        actions['a1_1_preference'] = (actions['a1_1'] + 1) / (actions['a1_-1'] + actions['a1_1'] + actions['a1_2'] + actions['a1_3'] + 1)
        actions['a1_2_preference'] = (actions['a1_2'] + 1) / (actions['a1_-1'] + actions['a1_1'] + actions['a1_2'] + actions['a1_3'] + 1)
        actions['a1_3_preference'] = (actions['a1_3'] + 1) / (actions['a1_-1'] + actions['a1_1'] + actions['a1_2'] + actions['a1_3'] + 1)

        actions['a2_-1_preference'] = (actions['a2_-1'] + 1) / (actions['a2_-1'] + actions['a2_1'] + actions['a2_2'] + 1)
        actions['a2_1_preference'] = (actions['a2_1'] + 1) / (actions['a2_-1'] + actions['a2_1'] + actions['a2_2'] + 1)
        actions['a2_2_preference'] = (actions['a2_2'] + 1) / (actions['a2_-1'] + actions['a2_1'] + actions['a2_2'] + 1)
        
        actions['a3_-1_preference'] = (actions['a3_-1'] + 1) / (actions['a3_-1'] + actions['a3_1'] + actions['a3_2'] + 1)
        actions['a3_1_preference'] = (actions['a3_1'] + 1) / (actions['a3_-1'] + actions['a3_1'] + actions['a3_2'] + 1)
        actions['a3_2_preference'] = (actions['a3_2'] + 1) / (actions['a3_-1'] + actions['a3_1'] + actions['a3_2'] + 1)

        actions = actions[['user_id', 'a1_-1_preference', 'a1_1_preference', 'a1_2_preference', 'a1_3_preference',
                            'a2_-1_preference', 'a2_1_preference', 'a2_2_preference',
                            'a3_-1_preference', 'a3_1_preference', 'a3_2_preference']]

        pickle.dump(actions, open(dump_path, 'w'))
    return actions

def get_cate_compete_feat(start_date, end_date):
    dump_path = './cache/cate_compete_feat_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        action_1 = get_actions_1()
        action_1 = action_1[(action_1.time >= start_date) & (action_1.time < end_date)]
        del action_1['brand'], action_1['model_id'], action_1['time']
        action_2 = get_actions_2()
        action_2 = action_2[(action_2.time >= start_date) & (action_2.time < end_date)]
        del action_2['brand'], action_2['model_id'], action_2['time']
        actions = pd.concat([action_1, action_2])
        del action_1, action_2
        action_3 = get_actions_3()
        action_3 = action_3[(action_3.time >= start_date) & (action_3.time < end_date)]
        del action_3['brand'], action_3['model_id'], action_3['time']
        actions = pd.concat([actions, action_3]) # type: pd.DataFrame
        del action_3

        actions['time'] = actions['time'].map(get_date)



        pickle.dump(actions, open(dump_path, 'w'))
    return actions

def get_user_product_info_accumulate_feat(start_date, end_date):
    return 0

def get_labels(start_date, end_date, test=False):
    dump_path = './cache/labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        action_1 = get_actions_1()
        action_1 = action_1[(action_1.time >= start_date) & (action_1.time < end_date)]
        del action_1['brand'], action_1['model_id'], action_1['time']
        action_2 = get_actions_2()
        action_2 = action_2[(action_2.time >= start_date) & (action_2.time < end_date)]
        del action_2['brand'], action_2['model_id'], action_2['time']
        actions = pd.concat([action_1, action_2])
        del action_1, action_2
        action_3 = get_actions_3()
        action_3 = action_3[(action_3.time >= start_date) & (action_3.time < end_date)]
        del action_3['brand'], action_3['model_id'], action_3['time']
        actions = pd.concat([actions, action_3]) # type: pd.DataFrame
        del action_3

        if test:
            actions = actions[actions['cate'] == 8]
        actions = actions[actions['type'] == 4]
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
        
        pickle.dump(actions, open(dump_path, 'w'))
    return actions

# def get_product_twice_buy_feat():

def make_test_set(train_start_date, train_end_date):
    dump_path = './cache/test_set_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        start_days = "2016-02-01"
        user = get_basic_user_feat()
        product = get_basic_product_feat()
        user_acc = get_accumulate_user_feat(start_days, train_end_date)
        product_acc = get_accumulate_product_feat(start_days, train_end_date)
        comment_acc = get_comments_product_feat(train_start_date, train_end_date)
        #labels = get_labels(test_start_date, test_end_date)

        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        actions = None
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_action_feat(start_days, train_end_date)
            else:
                actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                                   on=['user_id', 'sku_id'])

        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        actions = pd.merge(actions, product, how='left', on='sku_id')
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        #actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
        actions = actions.fillna(0)
        actions = actions[actions['cate'] == 8]

    users = actions[['user_id', 'sku_id']].copy()
    del actions['user_id']
    del actions['sku_id']
    return users, actions

def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, test=False):
    dump_path = './cache/train_set_%s_%s_%s_%s.pkl' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        # all user_product items in the special action_days
        action_days = 7
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=action_days)
        start_days = start_days.strftime('%Y-%m-%d')
        actions = get_action_feat(start_days, train_end_date) #initial actions

        # the basic info of the user
        start_days = "2016-01-31"
        actions = pd.merge(actions, get_basic_user_feat(), on=['user_id'], how='left')

        # the basic info of the product
        actions = pd.merge(actions, get_basic_product_feat(), on=['sku_id'], how='left')

        if test:  # if the data is userd to test, then get the special actions belong to cate 8
            actions = actions[actions['cate'] == 8]

        # the history actions of the user, or the consuming ability
        actions = pd.merge(actions, get_accumulate_user_feat(start_days, train_end_date), on=['user_id'], how='left')
        
        # the history interaction of the product, or the concern of the product
        actions = pd.merge(actions, get_accumulate_product_feat(start_days, train_end_date), on=['sku_id'], how='left')

        # the comment history info of the product
        actions = pd.merge(actions, get_comments_product_feat(train_end_date), on=['sku_id'], how='left')

        # the distance of user_product_touch to end_date
        actions = pd.merge(actions, get_user_product_touch_feat(start_days, train_end_date), on=['user_id', 'sku_id'], how='left')

        # the product cart feature
        actions = pd.merge(actions, get_product_cart_feat(start_days, train_end_date), on=['user_id', 'sku_id'], how='left')

        # the all actions about sku_id of user
        actions = pd.merge(actions, get_action_feat(start_days, train_end_date), on=['user_id', 'sku_id'], how='left')

        # the user reference for cate==8
        actions = pd.merge(actions, get_user_cate_feat(start_days, train_end_date), on=['user_id'], how='left')

        # the user active degree measured by days
        # actions = pd.merge(actions, get_user_active_feat(start_days, train_end_date), on=['user_id'], how='left')

        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        for i in (1, 2, 3, 5, 7):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                                   on=['user_id', 'sku_id'])

        for i in (1,3,5,7,10):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(hours=i)
            start_days = start_days.strftime('%Y-%m-%d %H:%M:%S')
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                                   on=['user_id', 'sku_id'])

        # the labels of the test_date
        labels = get_labels(test_start_date, test_end_date, test)
        if not test:
            actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])

        actions = actions.fillna(0)

    users = actions[['user_id', 'sku_id']].copy()
    if not test:
        labels = actions['label'].copy()
        del actions['label']
    del actions['user_id']
    del actions['sku_id']

    return users, actions, labels

def make_slide_train_set(train_start_date, train_end_date, test_start_date, test_end_date, step=7):
    dump_path = './cache/slide_train_set_%s_%s_%s_%s.pkl' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path))
    else:
        # all user_product items in the special action_days
        action_days = 7
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=action_days)
        start_days = start_days.strftime('%Y-%m-%d')
        actions = get_action_feat(start_days, train_end_date) #initial actions

    return users, actions, labels

def report(pred, label):

    actions = label
    result = pred

    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()

    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
            print(user_id)
    all_user_acc = 1.0 * pos / ( pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print('预测正确用户数：%s' % str(pos))
    print('预测错误用户数：%s' % str(neg))
    print('所有用户数：%s' % len(all_user_set))
    print '所有用户中预测购买用户的准确率为 ' + str(all_user_acc)
    print '所有用户中预测购买用户的召回率' + str(all_user_recall)

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
            print(user_item_pair)
    all_item_acc = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print('预测正确用户商品对数：%s' % str(pos))
    print('预测错误用户商品对数：%s' % str(neg))
    print('所有商品对数：%s' % len(all_user_item_pair))
    print '所有用户中预测购买商品的准确率为 ' + str(all_item_acc)
    print '所有用户中预测购买商品的召回率' + str(all_item_recall)
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print 'F11=' + str(F11)
    print 'F12=' + str(F12)
    print 'score=' + str(score)
    return score

if __name__ == '__main__':
    train_start_date = '2016-02-01'
    train_end_date = '2016-03-01'
    test_start_date = '2016-03-01'
    test_end_date = '2016-03-05'
    user, action, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    print user.head(10)
    print action.head(10)

# todo:
# user_cate_feat not finish
# product_info 
# comment_info
# cate_compete has_been_cate_buy cate_action_acc
# for acc
# user active days

# get_action_feat(start_days, train_end_date) yes
# product_twice_buy  yes
# user_twice_buy
# user_product_twice 
# user_product last action
# missing process in each feature   yes
# missing=-999.0


