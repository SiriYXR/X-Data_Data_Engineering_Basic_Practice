# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:test3.py
@time:2020/1/8 23:17
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASIZE=10

d={"A":pd.Series(np.random.randint(0,100,size=DATASIZE),index=list(range(1,DATASIZE+1))),
   "B":pd.Series(np.random.randint(0,100,size=DATASIZE),index=list(range(1,DATASIZE+1))),
   "C":pd.Series(np.random.randint(0,100,size=DATASIZE),index=list(range(1,DATASIZE+1))),
   "D":pd.Series(np.random.randint(0,100,size=DATASIZE),index=list(range(1,DATASIZE+1))),
   "E":pd.Series(np.random.randint(0,100,size=DATASIZE),index=list(range(1,DATASIZE+1)))
   }

df=pd.DataFrame(d)

print(df)

# 柱状图
df.plot(kind='bar')
plt.show()

# 散点图
df.plot(kind='scatter',x='A',y='B')
plt.show()

# 面积图
df.plot(kind='area')
plt.show()
