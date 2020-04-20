# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:xgb_modify.py
@time:2020/2/9 13:31
"""

import pandas as pd
from datetime import datetime
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=2, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['label'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['label'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['label'], dtrain_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.to_csv(r'./output_files/feat_importance.csv')
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

if __name__ == '__main__':
    start = datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))

    train = pd.read_csv(r'./prepared_dataset/train.csv')
    target='label'

    # 丢掉重要性低的特征
    # feat_imp=pd.read_csv(r'./output_files/xgb/09_2047_feat_importance_7735.csv')
    # drop_feature=list(feat_imp[feat_imp['importance'] < 1000]['feature_name'])

    # train.drop(drop_feature, axis=1, inplace=True)

    predictors = [x for x in train.columns if x not in ['label', 'User_id', 'Coupon_id', 'Date_received']]

    # xgb1 = XGBClassifier(
    #     learning_rate=0.01,
    #     n_estimators=20000,
    #     max_depth=5,
    #     min_child_weight=1,
    #     gamma=0.1,
    #     subsample=0.7,
    #     colsample_bylevel=0.7,
    #     colsample_bytree=0.7,
    #     objective='binary:logistic',
    #     nthread=4,
    #     scale_pos_weight=1,
    #     seed=27)
    #
    # #调优n_estimators
    # modelfit(xgb1, train, predictors)
    #
    # params=xgb1.get_params()
    # print(params)

    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'nthread': 4,
              'silent': 0,
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1.1,
              'gamma': 0.1,
              'lambda': 10,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.7,
              'scale_pos_weight': 1,
              'seed': 27}
    xgtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    cvresult = xgb.cv(params, xgtrain, num_boost_round=10000, nfold=2, metrics='auc', seed=27, callbacks=[
        xgb.callback.print_evaluation(show_stdv=False),
        xgb.callback.early_stop(50)
    ])
    num_round_best = cvresult.shape[0] - 1
    print('Best round num: ', num_round_best)

    # max_depth 和 min_weight 参数调优
    # param_test1 = {
    #     'max_depth': range(3, 10, 2),
    #     'min_child_weight': range(1, 6, 2)
    # }
    # gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.01, n_estimators=563, max_depth=5,
    #                                                 min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.7,colsample_bylevel=0.7,
    #                                                 objective='binary:logistic', nthread=4, scale_pos_weight=1,
    #                                                 seed=27),
    #                         param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    # gsearch1.fit(train[predictors], train[target])
    # print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

    # param_test2 = {
    #     'max_depth': [8, 9, 10],
    #     'min_child_weight': [4, 5, 6]
    # }
    # gsearch2 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.01, n_estimators=563, max_depth=5,
    #                                                 min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.7,colsample_bylevel=0.7,
    #                                                 objective='binary:logistic', nthread=4, scale_pos_weight=1,
    #                                                 seed=27),
    #                         param_grid=param_test2, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    # gsearch2.fit(train[predictors], train[target])
    # print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)

    # param_test2b = {
    #     'min_child_weight': [18,20]
    # }
    # gsearch2b = GridSearchCV(estimator=XGBClassifier(learning_rate=0.01, n_estimators=563, max_depth=9,
    #                                                  min_child_weight=6, gamma=0, subsample=0.9, colsample_bytree=0.7,colsample_bylevel=0.7,
    #                                                  objective='binary:logistic', nthread=4, scale_pos_weight=1,
    #                                                  seed=27),
    #                          param_grid=param_test2b, scoring='roc_auc', n_jobs=4,
    #                          iid=False, cv=5)
    #
    # gsearch2b.fit(train[predictors], train[target])
    #
    # print(gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_)

    # gamma 参数调优
    # param_test3 = {
    #     'gamma': [i / 10.0 for i in range(0, 5)]
    # }
    # gsearch3 = GridSearchCV(
    #     estimator=XGBClassifier(learning_rate=0.1, n_estimators=563, max_depth=3, min_child_weight=14, gamma=0,
    #                             subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4,
    #                             scale_pos_weight=1, seed=27), param_grid=param_test3, scoring='roc_auc', n_jobs=4,
    #     iid=False, cv=5)
    #
    # gsearch3.fit(train[predictors], train[target])
    # print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)

    # subsample 和 colsample_bytree 参数调优
    # param_test4 = {
    #     'subsample': [i / 10.0 for i in range(6, 10)],
    #     'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    # }
    #
    # gsearch4 = GridSearchCV(
    #     estimator=XGBClassifier(learning_rate=0.1, n_estimators=563, max_depth=3, min_child_weight=14, gamma=0.1,
    #                             subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4,
    #                             scale_pos_weight=1, seed=27), param_grid=param_test4, scoring='roc_auc', n_jobs=4,
    #     iid=False, cv=5)
    #
    # gsearch4.fit(train[predictors], train[target])
    # print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)

    # param_test5 = {
    #     'subsample': [i / 100.0 for i in range(65, 80, 5)],
    #     'colsample_bytree': [i / 100.0 for i in range(85, 100, 5)]
    # }
    #
    # gsearch5 = GridSearchCV(
    #     estimator=XGBClassifier(learning_rate=0.1, n_estimators=563, max_depth=3, min_child_weight=14, gamma=0.1,
    #                             subsample=0.7, colsample_bytree=0.9, objective='binary:logistic', nthread=4,
    #                             scale_pos_weight=1, seed=27), param_grid=param_test5, scoring='roc_auc', n_jobs=4,
    #     iid=False, cv=5)
    #
    # gsearch5.fit(train[predictors], train[target])
    # print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)

    # reg_alpha 参数调优
    # param_test6 = {
    #     'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    # }
    # gsearch6 = GridSearchCV(
    #     estimator=XGBClassifier(learning_rate=0.1, n_estimators=563, max_depth=3, min_child_weight=14, gamma=0.1,
    #                             subsample=0.7, colsample_bytree=0.9, objective='binary:logistic', nthread=4,
    #                             scale_pos_weight=1, seed=27), param_grid=param_test6, scoring='roc_auc', n_jobs=4,
    #     iid=False, cv=5)
    #
    # gsearch6.fit(train[predictors], train[target])
    # print(gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_)

    # param_test7 = {
    #     'reg_alpha': [5e-5,1e-4,5e-4,1e-3,5e-3]
    # }
    # gsearch7 = GridSearchCV(
    #     estimator=XGBClassifier(learning_rate=0.1, n_estimators=563, max_depth=3, min_child_weight=14, gamma=0.1,
    #                             subsample=0.7, colsample_bytree=0.9, objective='binary:logistic', nthread=4,
    #                             scale_pos_weight=1, seed=27), param_grid=param_test7, scoring='roc_auc', n_jobs=4,
    #     iid=False, cv=5)
    #
    # gsearch7.fit(train[predictors], train[target])
    # print(gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_)

    # colsample_bylevel 参数调优
    # param_test8 = {
    #     'colsample_bylevel': [i / 100.0 for i in range(60, 100, 10)]
    # }
    #
    # gsearch8 = GridSearchCV(
    #     estimator=XGBClassifier(learning_rate=0.1, n_estimators=563, max_depth=3, min_child_weight=14, gamma=0.1,
    #                             subsample=0.7, colsample_bytree=0.9, objective='binary:logistic', nthread=4,
    #                             scale_pos_weight=1,reg_alpha=0.0005, seed=27), param_grid=param_test8, scoring='roc_auc', n_jobs=4,
    #     iid=False, cv=5)
    #
    # gsearch8.fit(train[predictors], train[target])
    # print(gsearch8.grid_scores_, gsearch8.best_params_, gsearch8.best_score_)

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %d min' % (int((datetime.now() - start).seconds) / 60))