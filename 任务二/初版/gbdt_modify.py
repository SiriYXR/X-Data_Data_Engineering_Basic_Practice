# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:gbdt_modify.py
@time:2020/2/11 18:50
"""

import pandas as pd
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    start = datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))

    train = pd.read_csv(r'./prepared_dataset/train.csv')
    validate = pd.read_csv("./prepared_dataset/validate.csv")

    big_train = pd.concat([train, validate], axis=0)
    y = big_train['label']
    X = big_train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)

    # 迭代次数（n_estimator）
    # param_test1 = {'n_estimators':range(20,81,10)}
    # grid_search1 = GridSearchCV(
    #     estimator=GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,
    #                                          min_samples_split=400, min_samples_leaf=18, max_depth=6,
    #                                          max_features='sqrt', subsample=0.8, random_state=10,verbose=2),
    #     param_grid=param_test1,
    #     scoring='roc_auc',
    #     iid=False,
    #     cv=5
    # )
    # grid_result1 = grid_search1.fit(X, y)
    # ##打印结果
    # print("Best: %f using %s" % (grid_result1.best_score_, grid_result1.best_params_))
    # means = grid_result1.cv_results_['mean_test_score']
    # params = grid_result1.cv_results_['params']
    # for mean, param in zip(means, params):
    #     print("mean:  %f  , params:  %r" % (mean, param))

    # 决策树最大深度（max_depth）& 内部节点划分所需最小样本数（min_samples_split）
    #param_test2 = {'max_depth': range(3, 8, 2), 'min_samples_split': range(100, 801, 200)}
    #param_test2 = {'max_depth': [2,3,4], 'min_samples_split': [400,500,600]}
    # param_test2 = {'max_depth': [1,2],'min_samples_split': [500]}
    # grid_search2 = GridSearchCV(
    #     estimator=GradientBoostingClassifier(learning_rate=0.1,
    #                                          n_estimators=70, min_samples_leaf=18,max_depth=6,
    #                                          max_features='sqrt', subsample=0.8, random_state=10,verbose=2),
    #     param_grid=param_test2,
    #     scoring='roc_auc',
    #     iid=False,
    #     cv=5
    # )
    # grid_result2 = grid_search2.fit(X, y)
    # ##打印结果
    # print("Best: %f using %s" % (grid_result2.best_score_, grid_result2.best_params_))
    # means = grid_result2.cv_results_['mean_test_score']
    # params = grid_result2.cv_results_['params']
    # for mean, param in zip(means, params):
    #     print("mean:  %f  , params:  %r" % (mean, param))

    # 内部节点再划分所需最小样本数（min_samples_split） & 叶子节点最少样本数（min_samples_leaf）
    # param_test3 = {'min_samples_split': range(400, 801, 200), 'min_samples_leaf': range(10, 41, 10)}
    # grid_search3 = GridSearchCV(
    #     estimator=GradientBoostingClassifier(learning_rate=0.1,
    #                                          n_estimators=70, max_depth=6,
    #                                          max_features='sqrt', subsample=0.8, random_state=10,verbose=2),
    #     param_grid=param_test3,
    #     scoring='roc_auc',
    #     iid=False,
    #     cv=5
    # )
    # grid_result3 = grid_search3.fit(X, y)
    # ##打印结果
    # print("Best: %f using %s" % (grid_result3.best_score_, grid_result3.best_params_))
    # means = grid_result3.cv_results_['mean_test_score']
    # params = grid_result3.cv_results_['params']
    # for mean, param in zip(means, params):
    #     print("mean:  %f  , params:  %r" % (mean, param))

    # 最大特征数（max_features）
    param_test4 = {'max_features': range(7, 20, 2)}
    # param_test4 = {'max_features': [1,2,3]}
    grid_search4 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1,
                                             n_estimators=70, max_depth=6, min_samples_leaf=18,
                                             min_samples_split=400, subsample=0.8, random_state=10,verbose=2),
        param_grid=param_test4,
        scoring='roc_auc',
        iid=False,
        cv=5
    )
    grid_result4 = grid_search4.fit(X, y)
    ##打印结果
    print("Best: %f using %s" % (grid_result4.best_score_, grid_result4.best_params_))
    means = grid_result4.cv_results_['mean_test_score']
    params = grid_result4.cv_results_['params']
    for mean, param in zip(means, params):
        print("mean:  %f  , params:  %r" % (mean, param))

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %d min' % (int((datetime.now() - start).seconds) / 60))