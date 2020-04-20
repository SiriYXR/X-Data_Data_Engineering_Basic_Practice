# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:gbdt_train.py
@time:2020/2/11 17:09
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


def monitor(i,self,localv):

    return False

def model_gbdt(train, validate, test):
    big_train = pd.concat([train, validate], axis=0)

    y = big_train['label']
    X = big_train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1)

    test_X = test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1)

    # 训练
    gbr = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=6, min_samples_split=400,
                                     min_samples_leaf=18, random_state=10, verbose=2)
    gbr.fit(X, y)

    # 预测
    y_pred=gbr.predict(X)
    y_proba=gbr.predict_proba(X)[:,1]
    predict=gbr.predict_proba(test_X)[:,1]
    print("Accuracy:%.4f" % metrics.accuracy_score(y, y_pred))
    print("AUC Score(Train):%f\n" % metrics.roc_auc_score(y, y_proba))

    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)

    feat_imp = pd.Series(gbr.feature_importances_, X.columns).sort_values(ascending=False)

    return result,feat_imp


if __name__ == "__main__":
    start = datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))

    train = pd.read_csv(r'./prepared_dataset/train.csv')
    validate = pd.read_csv("./prepared_dataset/validate.csv")
    test = pd.read_csv("./prepared_dataset/test.csv")

    # 训练
    result,feat_importance= model_gbdt(train, validate, test)

    # 保存
    #result.to_csv(r'./output_files/gbdt/' + datetime.now().strftime('%d_%H%M') + '_test.csv', index=False, header=None)
    #feat_importance.to_csv(r'./output_files/gbdt/' + datetime.now().strftime('%d_%H%M') + '_feat_importance.csv')

    print(feat_importance)
    # feat_importance.plot(kind='bar')
    # plt.show()

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %d min' % (int((datetime.now() - start).seconds) / 60))