# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:creatData.py
@time:2020/1/10 20:22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
   data=np.random.rand(100)*100

   data.resize((50,2))

   plt.scatter(data[:,0],data[:,1])
   plt.show()

   d={'X':pd.Series(data[:,0]),
      'Y':pd.Series(data[:,1])}

   df=pd.DataFrame(d)

   df.to_csv("./testdata.csv", encoding="utf-8-sig", header=True, index=False)