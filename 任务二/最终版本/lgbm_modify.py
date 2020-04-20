# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:lgbm_modify.py
@time:2020/2/10 15:28
"""

from datetime import datetime
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    start = datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))

    train = pd.read_csv(r'./prepared_dataset/train.csv')
    validate = pd.read_csv(r'./prepared_dataset/validate.csv')

    big_train = pd.concat([train, validate], axis=0)

    y = big_train['label']
    X = big_train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)

    # # n_estimators调优
    # params = {'boosting_type': 'gbdt',
    #           'objective': 'binary',
    #           'learning_rate': 0.05,
    #           'metric': {'l2', 'auc'},
    #           'num_leaves': 18,
    #           'max_depth': 13,
    #           'subsample': 0.8,
    #           'min_child_samples': 20,
    #           'min_child_weight': 0.001,
    #           'colsample_bytree': 0.8,
    #           'feature_fraction': 0.82,  # 建树的特征选择比例
    #           'bagging_fraction': 1.0,  # 建树的样本采样比例
    #           'reg_alpha': 0.58,
    #           'reg_lambda': 0.56,
    #           'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    #           'verbose': -1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    #           }
    #
    # data_train = lgb.Dataset(X, y, free_raw_data=False)
    # cv_results = lgb.cv(
    #     params, data_train, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='auc',
    #     early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)
    #
    # print('best n_estimators:', len(cv_results['auc-mean']))
    # print('best cv score:', cv_results['auc-mean'][-1])


    # max_depth 和 num_leaves 调优
    # params_test1 = {'max_depth': range(3, 8, 2), 'num_leaves': range(20, 170, 30)
    #                 }
    params_test1 = {'max_depth': [1,2,3,4]
                    }
    model_lgb = lgb.LGBMRegressor(objective='binary', num_leaves=18,
                                  learning_rate=0.1, n_estimators=170, max_depth=13,subsample=0.8,colsample_bytree=0.8,
                                  metric='auc', bagging_fraction=1.0, feature_fraction=0.82,bagging_freq=5,verbose=-1)

    gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='roc_auc', cv=5,
                            verbose=-1, n_jobs=4)
    gsearch1.fit(X, y)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

    # params_test2 = {'max_depth': [12,13,14], 'num_leaves': [17,18,19]}
    # gsearch2 = GridSearchCV(estimator=model_lgb, param_grid=params_test2, scoring='roc_auc', cv=5,
    #                         verbose=-1, n_jobs=4)
    # gsearch2.fit(X, y)
    # print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)


    # min_data_in_leaf(min_child_samples) 和 min_sum_hessian_in_leaf(min_child_weight) 调优
    # params_test3 = {'min_child_samples': [18, 19, 20, 21, 22], 'min_child_weight': [0.001, 0.002]
    #                 }
    # model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=18, learning_rate=0.1, n_estimators=340,
    #                               max_depth=13, subsample=0.8, colsample_bytree=0.8, metric='auc', bagging_fraction=0.8,
    #                               feature_fraction=0.9, bagging_freq=5, verbose=-1)
    # gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='roc_auc', cv=5,
    #                         verbose=-1, n_jobs=4)
    # gsearch3.fit(X, y)
    # print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)


    # feature_fraction 和 bagging_fraction 调优
    # params_test4 = {'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9], 'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
    #                 }
    # model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=18, learning_rate=0.1, n_estimators=340,
    #                                max_depth=13, subsample=0.8, colsample_bytree=0.8, metric='auc', bagging_fraction=0.8,
    #                                feature_fraction=0.9,min_child_samples=20,min_child_weight=0.001, bagging_freq=5, verbose=-1)
    # gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='roc_auc', cv=5,
    #                         verbose=1, n_jobs=4)
    # gsearch4.fit(X, y)
    # print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)

    # params_test5 = {'feature_fraction': [0.72, 0.75, 0.78, 0.8, 0.82, 0.85, 0.88 ]
    #                 }
    # model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=18, learning_rate=0.1, n_estimators=340,
    #                               max_depth=13, subsample=0.8, colsample_bytree=0.8, metric='auc', bagging_fraction=1.0,
    #                               feature_fraction=0.9, min_child_samples=20, min_child_weight=0.001, bagging_freq=5,
    #                               verbose=-1)
    # gsearch5 = GridSearchCV(estimator=model_lgb, param_grid=params_test5, scoring='roc_auc', cv=5,
    #                         verbose=1, n_jobs=4)
    # gsearch5.fit(X, y)
    # print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)

    # 正则化参数
    # params_test6 = {'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
    #                 'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]
    #                 }
    # params_test6 = {'reg_alpha': [0.4,0.5,0.6,0.7],
    #                 'reg_lambda': [0.4,0.5,0.6,0.7]
    #                 }
    # params_test6 = {'reg_alpha': [0.56, 0.58, 0.6, 0.62,65],
    #                 'reg_lambda': [0.56, 0.58, 0.6, 0.62,65]
    #                 }
    # model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=18, learning_rate=0.1, n_estimators=340,
    #                               max_depth=13, subsample=0.8, colsample_bytree=0.8, metric='auc', bagging_fraction=1.0,
    #                               feature_fraction=0.82, min_child_samples=20, min_child_weight=0.001, bagging_freq=5,
    #                               verbose=-1)
    # gsearch6 = GridSearchCV(estimator=model_lgb, param_grid=params_test6, scoring='roc_auc', cv=5,
    #                         verbose=1, n_jobs=4)
    # gsearch6.fit(X, y)
    # print(gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_)


    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %d min' % (int((datetime.now() - start).seconds) / 60))