# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:test1.py
@time:2020/1/8 21:09
"""

import numpy as np
import pandas as pd

DATASIZE=100000

d={"1_白细胞量":pd.Series(np.random.rand(DATASIZE)*10,index=list(range(1,DATASIZE+1))),
   "2_红细胞量":pd.Series(np.random.rand(DATASIZE)*10,index=list(range(1,DATASIZE+1))),
   "3_血小板量":pd.Series(np.random.rand(DATASIZE)*10,index=list(range(1,DATASIZE+1))),
   "4_是否患癌":pd.Series(np.random.randint(0,2,size=DATASIZE),index=list(range(1,DATASIZE+1))),
   "5_预测标签":pd.Series(np.random.randint(0,2,size=DATASIZE),index=list(range(1,DATASIZE+1)))
   }

df=pd.DataFrame(d)

print(df)

# 混淆矩阵:TP([1][1])、FN([1][0])、FP([0][1])、TN([0][0])
CM=[[0,0],[0,0]]

for i in range(0,DATASIZE):
   CM[df.iloc[i,3]][df.iloc[i,4]]+=1

print("\nTP:%d \t FN:%d \nFP:%d \t TN:%d\n"%(CM[1][1],CM[1][0],CM[0][1],CM[0][0]))

# 查准率
P=(CM[1][1])/(CM[1][1]+CM[0][1])

# 查全率
R=(CM[1][1])/(CM[1][1]+CM[1][0])

print("查准率:%.3f \t 查全率:%.3f \n"%(P,R))

# F1-score
F1=(2*P*R)/(P+R)

print("F1-score:%.3f"%(F1))