# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:test2.py
@time:2020/1/8 21:07
"""

# for 循环

s = 0

for i in range(1,1001):
    s += i

print("for 循环:",s)

# 列表推导式

print("列表推导式:",sum([ x for x in range(1,1001)]))