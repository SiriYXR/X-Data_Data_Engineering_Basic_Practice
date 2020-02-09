# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:test5.py
@time:2020/1/11 16:27
"""
import os
import pandas as pd
from pyecharts.charts import Bar
from pyecharts import options as opts

def task1():
    # 读取数据集
    off_test=pd.read_csv("ccf_offline_stage1_test_revised.csv")
    # 共有多少条记录
    record_count=off_test.shape[0]
    # 共有多少条优惠券的领取记录
    received_count = off_test['Date_received'].count()
    # 共有多少种不同的优惠券
    coupon_count = len(off_test['Coupon_id'].value_counts())
    # 共有多少个用户
    user_count = len(off_test['User_id'].value_counts())
    # 共有多少个商家
    merchant_count = len(off_test['Merchant_id'].value_counts())
    # 最早领券时间
    min_received = str(int(off_test['Date_received'].min()))
    # 最晚领券时间
    max_received = str(int(off_test['Date_received'].max()))

    print('record_count', record_count)
    print('received_count', received_count)
    print('coupon_count', coupon_count)
    print('user_count', user_count)
    print('merchant_count', merchant_count)
    print('min_received', min_received)
    print('max_received', max_received)

def task2():
    # 源数据
    data = pd.read_csv('ccf_offline_stage1_test_revised.csv')

    # 复制data数据集为其做一下预处理方便绘图
    offline = data.copy()

    # 将Distance的空值填充为-1
    offline['Distance'].fillna(-1, inplace=True)
    # 将领券时间转为时间类型
    offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
    # 将折扣券转为折扣率
    offline['discount_rate'] = offline['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))

    # 为数据打标
    offline['label'] = list(1 for x in range(offline.shape[0]))
    # 添加优惠券是否为满减类型
    offline['is_manjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    # 领券时间为周几
    offline['weekday_received'] = offline['date_received'].apply(lambda x: x.isoweekday())

    # 为offline数据添加一个received_month即领券月份
    offline['received_month'] = offline['date_received'].apply(lambda x: x.month)

    path = './tmp/task2_output'
    if not os.path.exists(path):
        os.makedirs(path)
    offline.to_csv(path + '/task2_output.csv', index=False)

    print(offline)

def task3():
    # 查看pyecharts版本，本节代码只适合1.x版本
    # print(pyecharts.__version__)

    # 源数据
    data = pd.read_csv('ccf_offline_stage1_test_revised.csv')

    # 复制data数据集为其做一下预处理方便绘图
    offline = data.copy()

    # 将Distance的空值填充为-1
    offline['Distance'].fillna(-1, inplace=True)
    # 将领券时间转为时间类型
    offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
    # 将折扣券转为折扣率
    offline['discount_rate'] = offline['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))

    # 为数据打标
    offline['label'] = list(1 for x in range(offline.shape[0]))
    # 添加优惠券是否为满减类型
    offline['is_manjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    # 领券时间为周几
    offline['weekday_received'] = offline['date_received'].apply(lambda x: x.isoweekday())

    # 为offline数据添加一个received_month即领券月份
    offline['received_month'] = offline['date_received'].apply(lambda x: x.month)


    ##########################
    # 选取领券日期不为空的数据
    df_1 = offline[offline['Date_received'].notnull()]
    # 以Date_received为分组目标并统计优惠券的数量
    tmp = df_1.groupby('Date_received', as_index=False)['Coupon_id'].count()


    # 建立柱状图
    bar_1 = (
        Bar(
            init_opts=opts.InitOpts(width='1500px', height='600px')
        )
            .add_xaxis(list(tmp['Date_received']))
            .add_yaxis('', list(tmp['Coupon_id']))
            .set_global_opts(
            title_opts=opts.TitleOpts(title='每天被领券的数量'),  # title
            legend_opts=opts.LegendOpts(is_show=True),  # 显示ToolBox
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60), interval=1),  # 旋转60度
        )
            .set_series_opts(
            opts.LabelOpts(is_show=False),  # 显示值大小
            markline_opts=opts.MarkLineOpts(
                data=[
                    opts.MarkLineItem(type_='max', name='最大值') # 标注最大值
                ]
            )
        )
    )

    ######################
    # 消费距离柱状图
    # 统计各类距离的消费次数
    import collections
    dis = offline[offline['Distance'] != -1]['Distance'].values
    dis = dict(collections.Counter(dis))

    x = list(dis.keys())
    y = list(dis.values())

    # 建立柱状图
    bar_2 = (
        Bar()
            .add_xaxis(x)
            .add_yaxis('', y)
            .set_global_opts(
            title_opts=opts.TitleOpts(title='用户消费距离统计'),  # title
        )
    )

    path = './tmp/task2_output'
    if not os.path.exists(path):
        os.makedirs(path)
    # render会生成本地HTML文件，默认在当前目录生成render.html文件
    # bar_1.render(path + '/bar_1.html')
    # bar_2.render(path + '/bar_2.html')
    # bar_3.render(path + '/bar_3.html')
    # bar_4.render(path + '/bar_4.html')
    pass

if __name__ == "__main__":
    task1()

    pass