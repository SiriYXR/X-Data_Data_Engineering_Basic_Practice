# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:xgb_train.py
@time:2020/2/9 13:31
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')  # 不显示警告


def model_xgb(train, test):
    """xgb模型

    """
    # xgb参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'nthread': 4,
              'silent': 0,
              'eta': 0.01,
              'max_depth': 9,
              'min_child_weight': 19,
              'gamma': 0,
              'lambda': 1,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.9,
              'scale_pos_weight': 1,
              'seed': 27}


    # 数据集
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    # 训练
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=3500, evals=watchlist, early_stopping_rounds=50)
    # 预测
    predict = model.predict(dtest)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    # 特征重要性
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)
    # 返回
    return result, feat_importance


if __name__ == '__main__':
    start = datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))

    train = pd.read_csv(r'./prepared_dataset/train.csv')
    validate = pd.read_csv("./prepared_dataset/validate.csv")
    test = pd.read_csv("./prepared_dataset/test.csv")

    # 丢掉重要性低的特征
    # feat_imp=pd.read_csv(r'./output_files/xgb/09_2047_feat_importance_7735.csv')
    # drop_feature=list(feat_imp[feat_imp['importance'] < 1000]['feature_name'])
    #
    # train.drop(drop_feature, axis=1, inplace=True)
    # validate.drop(drop_feature, axis=1, inplace=True)
    # test.drop(drop_feature, axis=1, inplace=True)

    # 训练
    big_train = pd.concat([train, validate], axis=0)
    result, feat_importance = model_xgb(big_train, test)
    # 保存
    result.to_csv(r'./output_files/xgb/'+datetime.now().strftime('%d_%H%M')+'_test.csv', index=False, header=None)
    feat_importance.to_csv(r'./output_files/xgb/'+datetime.now().strftime('%d_%H%M')+'_feat_importance.csv')

    print(feat_importance)
    feat_importance.plot(kind='bar')
    plt.show()

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %d min' % (int((datetime.now() - start).seconds)/60))