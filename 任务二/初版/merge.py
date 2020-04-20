# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:merge.py
@time:2020/2/9 20:39
"""

import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    start = datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))

    test1 = pd.read_csv("./output_files/gbdt/12_1744_test_7801.csv")
    test2 = pd.read_csv("./output_files/lgbm/11_1149_test_7824.csv")

    test3=test1.copy()
    for i in range(40,60):
        test3.iloc[:,3]=test1.iloc[:,3]*(i/100.0)+test2.iloc[:,3]*(1-(i/100.0))

        test3.to_csv('./output_files/merge/' + datetime.now().strftime('%d_%H%M') + '_merge_'+str(i)+'.csv',index=False)

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %d min' % (int((datetime.now() - start).seconds)/60))