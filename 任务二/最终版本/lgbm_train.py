# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:lgbm_train.py
@time:2020/2/10 13:59
"""

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb


def model_lgbm(train, validate, test):

    y = train['label']
    X = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'],axis=1)

    val_y=validate['label']
    val_X=validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'],axis=1)

    test_X=test.drop(['User_id', 'Coupon_id', 'Date_received'],axis=1)

    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X, y, free_raw_data=False)
    lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train, free_raw_data=False)

    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'binary',  # 目标函数
        'metric': 'auc',  # 评估函数
        'max_depth': 3,
        'num_leaves': 18,  # 叶子节点数
        'learning_rate': 0.1,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    # 训练 cv and train
    print('Start training...')
    gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=lgb_eval, early_stopping_rounds=50)

    # 保存模型到文件
    # print('Save model...')
    # gbm.save_model('./model/lgbm/' + datetime.now().strftime('%d_%H%M') + '_model.txt')

    # 预测
    print('Start predicting...')
    predict=gbm.predict(test_X, num_iteration=gbm.best_iteration)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)

    # 特征重要性
    feat_importance = pd.DataFrame(X.columns.tolist(), columns=['feature_name'])
    feat_importance['importance'] = list(gbm.feature_importance())
    feat_importance = feat_importance.sort_values(by='importance', ascending=False)

    return result,feat_importance

def model_lgbm2(train, validate, test):

    y = train['label']
    X = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'],axis=1)

    val_y=validate['label']
    val_X=validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'],axis=1)

    test_X=test.drop(['User_id', 'Coupon_id', 'Date_received'],axis=1)

    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X, y, free_raw_data=False)
    lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train, free_raw_data=False)

    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'binary',  # 目标函数
        'metric': 'auc',  # 评估函数
        'num_leaves': 42,  # 叶子节点数
        'learning_rate': 0.01,  # 学习速率
        'feature_fraction': 0.8,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k意味着每k次迭代执行bagging
        'max_depth': 9,
        'min_child_weight': 0.001,
        'min_data_in_leaf': 440,
        'seed': 34
    }

    # 训练 cv and train
    print('Start training...')
    gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=lgb_eval, early_stopping_rounds=50)

    # 保存模型到文件
    # print('Save model...')
    # gbm.save_model('./model/lgbm/' + datetime.now().strftime('%d_%H%M') + '_model.txt')

    # 预测
    print('Start predicting...')
    predict=gbm.predict(test_X, num_iteration=gbm.best_iteration)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)

    # 特征重要性
    feat_importance = pd.DataFrame(X.columns.tolist(), columns=['feature_name'])
    feat_importance['importance'] = list(gbm.feature_importance())
    feat_importance = feat_importance.sort_values(by='importance', ascending=False)

    return result,feat_importance

def model_lgbm3(train, validate, test):

    y = train['label']
    X = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'],axis=1)

    val_y=validate['label']
    val_X=validate.drop(['User_id', 'Coupon_id', 'Date_received', 'label'],axis=1)

    test_X=test.drop(['User_id', 'Coupon_id', 'Date_received'],axis=1)

    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X, y, free_raw_data=False)
    lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train, free_raw_data=False)

    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'binary',  # 目标函数
        'metric': 'auc',  # 评估函数
        'max_depth': 9,
        'num_leaves': 18,  # 叶子节点数
        'learning_rate': 0.1,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    # 训练 cv and train
    print('Start training...')
    gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=lgb_eval, early_stopping_rounds=50)

    # 保存模型到文件
    # print('Save model...')
    # gbm.save_model('./model/lgbm/' + datetime.now().strftime('%d_%H%M') + '_model.txt')

    # 预测
    print('Start predicting...')
    predict=gbm.predict(test_X, num_iteration=gbm.best_iteration)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)

    # 特征重要性
    feat_importance = pd.DataFrame(X.columns.tolist(), columns=['feature_name'])
    feat_importance['importance'] = list(gbm.feature_importance())
    feat_importance = feat_importance.sort_values(by='importance', ascending=False)

    return result,feat_importance

def model_lgbm_novalid(train, validate, test):
    big_train = pd.concat([train, validate], axis=0)

    y = big_train['label']
    X = big_train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)

    test_X=test.drop(['User_id', 'Coupon_id', 'Date_received'],axis=1)

    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X, y, free_raw_data=False)

    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'binary',  # 目标函数
        'metric': 'auc',  # 评估函数
        'max_depth': 3,
        'num_leaves': 18,  # 叶子节点数
        'learning_rate': 0.1,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    # 训练 cv and train
    print('Start training...')
    gbm = lgb.train(params, lgb_train, num_boost_round=180)

    # # 保存模型到文件
    # print('Save model...')
    # gbm.save_model('./model/lgbm/' + datetime.now().strftime('%d_%H%M') + '_model.txt')

    # 预测
    print('Start predicting...')
    predict=gbm.predict(test_X, num_iteration=gbm.best_iteration)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)

    # 特征重要性
    feat_importance = pd.DataFrame(X.columns.tolist(), columns=['feature_name'])
    feat_importance['importance'] = list(gbm.feature_importance())
    feat_importance = feat_importance.sort_values(by='importance', ascending=False)

    return result,feat_importance

def model_lgbm_novalid2(train, validate, test):
    big_train = pd.concat([train, validate], axis=0)

    y = big_train['label']
    X = big_train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)

    test_X=test.drop(['User_id', 'Coupon_id', 'Date_received'],axis=1)

    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X, y, free_raw_data=False)

    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'binary',  # 目标函数
        'metric': {'l2', 'auc'},  # 评估函数
        'max_depth': 3,
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1,# <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    # 训练 cv and train
    print('Start training...')
    gbm = lgb.train(params, lgb_train, num_boost_round=940)

    # # 保存模型到文件
    # print('Save model...')
    # gbm.save_model('./model/lgbm/' + datetime.now().strftime('%d_%H%M') + '_model.txt')

    # 预测
    print('Start predicting...')
    predict=gbm.predict(test_X, num_iteration=gbm.best_iteration)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)

    # 特征重要性
    feat_importance = pd.DataFrame(X.columns.tolist(), columns=['feature_name'])
    feat_importance['importance'] = list(gbm.feature_importance())
    feat_importance = feat_importance.sort_values(by='importance', ascending=False)

    return result,feat_importance

if __name__ == "__main__":
    start = datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))

    train = pd.read_csv(r'./prepared_dataset/train.csv')
    validate = pd.read_csv("./prepared_dataset/validate.csv")
    test = pd.read_csv("./prepared_dataset/test.csv")

    # 丢掉重要性低的特征
    # feat_imp = pd.read_csv(r'./output_files/xgb/09_2047_feat_importance_7735.csv')
    # drop_feature = list(feat_imp[feat_imp['importance'] < 1000]['feature_name'])
    #
    # train.drop(drop_feature, axis=1, inplace=True)
    # validate.drop(drop_feature, axis=1, inplace=True)
    # test.drop(drop_feature, axis=1, inplace=True)

    # 训练
    # result,feat_importance= model_lgbm(train, validate, test)
    result, feat_importance = model_lgbm2(train, validate, test)

    # 保存
    result.to_csv(r'./output_files/lgbm/' + datetime.now().strftime('%d_%H%M') + '_test.csv', index=False, header=None)
    # feat_importance.to_csv(r'./output_files/lgbm/' + datetime.now().strftime('%d_%H%M') + '_feat_importance.csv')

    # print(feat_importance)
    # feat_importance.plot(kind='bar')
    # plt.show()

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %d min' % (int((datetime.now() - start).seconds) / 60))