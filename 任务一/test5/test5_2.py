# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:test5_2.py
@time:2020/1/11 20:55
"""

import os
import pandas as pd
from pyecharts.charts import Bar,Pie
from pyecharts import options as opts
import collections

if __name__ == "__main__":

    # 源数据
    data = pd.read_csv('ccf_offline_stage1_test_revised.csv')

    # 复制data数据集为其做一下预处理方便绘图
    offline = data.copy()

    # 将Distance的空值填充为-1
    # offline['Distance'].fillna(-1, inplace=True) # 原表中的null无法使用fillna函数填充，所以直接替换
    offline.ix[offline['Distance'] == 'null', 'Distance'] = -1

    # 将领券时间转为时间类型
    offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
    # 将折扣券转为折扣率
    offline['discount_rate'] = offline['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    # 领券时间为周几
    offline['weekday_received'] = offline['date_received'].apply(lambda x: x.isoweekday())
    # 添加优惠券是否为满减类型
    offline['is_manjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)

    print(offline)
    offline.to_csv("./tmp/task2_output/task2_output.csv")


##########################
    # 选取领券日期不为空的数据
    df_1 = offline[offline['Date_received'].notnull()]
    # 以Date_received为分组目标并统计优惠券的数量
    tmp = df_1.groupby('Date_received', as_index=False)['Coupon_id'].count()

    x=list(tmp['Date_received'])
    y=list(tmp['Coupon_id'])

    # 建立柱状图
    bar_1 = (
        Bar(
            init_opts=opts.InitOpts(width='1500px', height='600px')
        )
            .add_xaxis(x)
            .add_yaxis('', y)
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

######################
    # 用户一周内各天领卷数量累计
    x=["周一","周二","周三","周四","周五","周六","周日"]
    y=[]

    for i in range(1,8):
        y.append(offline[offline['weekday_received'] == i]['weekday_received'].count())

    # 建立柱状图
    bar_3 = (
        Bar()
            .add_xaxis(x)
            .add_yaxis('', y)
            .set_global_opts(
            title_opts=opts.TitleOpts(title='用户一周内各天领卷数量累计'),  # title
        )
    )

######################
    # 优惠卷类型比例
    v1=['折扣','满减']
    v2=list(offline[offline['Date_received'].notnull()]['is_manjian'].value_counts(True))

    pie_1=(
        Pie()
        .add("",[list(v) for v in zip(v1,v2)])
        .set_global_opts(title_opts={"text":"各类优惠卷数量占比饼图"})
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    )

    path = './tmp/task2_output'
    if not os.path.exists(path):
        os.makedirs(path)
    # render会生成本地HTML文件，默认在当前目录生成render.html文件
    bar_1.render(path + '/bar_1.html')
    bar_2.render(path + '/bar_2.html')
    bar_3.render(path + '/bar_3.html')
    pie_1.render(path + '/pie_1.html')