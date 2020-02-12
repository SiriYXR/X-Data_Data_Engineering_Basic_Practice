# -*- coding:utf-8 -*-
"""
@author:SiriYang
@file:dataset_process.py
@time:2020/2/9 13:30
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')  # 不显示警告


def prepare(dataset):
    """数据预处理

    1.折扣处理:
        判断折扣是“满减”(如10:1)还是“折扣率”(0.9), 新增一列'is_manjian'表示该信息;
        将“满减”折扣转换为“折扣率”形式(如10:1转换为0.9), 新增一列'discount_rate'表示该信息;
        得到“满减”折扣的最低消费(如折扣10:1的最低消费为10), 新增一列'min_cost_of_manjian'表示该信息;
        将没有优惠卷的满减字段设为1，并且新增一个字段‘null_coupon’表示该信息；
    2.距离处理:
        将空距离填充为-1(区别于距离0,1,2,3,4,5,6,7,8,9,10);
        判断是否为空距离, 新增一列'null_distance'表示该信息;
    3.时间处理(方便计算时间差):
        将'Date_received'列中int或float类型的元素转换成datetime类型, 新增一列'date_received'表示该信息;
        将'Date'列中int类型的元素转换为datetime类型, 新增一列'date'表示该信息;

    Args:
        dataset: off_train和off_test, DataFrame类型的数据集包含属性'User_id', 'Merchant_id', 'Coupon_id',
                'Discount_rate', 'Distance', 'Date_received', 'Date'(off_test没有'Date'属性);

    Returns:
        data: 预处理后的DataFrame类型的数据集.
    """
    # 源数据

    # 先过滤掉没有领取优惠卷的数据
    data = dataset[~(dataset['Discount_rate'].isin(['null']))]

    print("折扣处理...")
    # 折扣处理
    # Discount_rate是否为满减
    data['is_manjian'] = data['Discount_rate'].map(
        lambda x: 1 if ':' in str(x) else 0)
    # 满减全部转换为折扣率,null转换为1
    data['discount_rate'] = data['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    # 满减的最低消费
    data['min_cost_of_manjian'] = data['Discount_rate'].map(lambda x: -1 if ':' not in str(x) else
    int(str(x).split(':')[0]))

    print("距离处理...")
    # 距离处理
    # 空距离填充为-1
    data['Distance'].fillna(-1, inplace=True)
    data.ix[data['Distance'] == 'null', 'Distance'] = -1
    data['Distance'] = data['Distance'].map(int)

    # 判断是否是空距离
    data['null_distance'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)

    print("时间处理...")
    # 时间处理
    #  float类型转换为datetime类型
    data['date_received'] = pd.to_datetime(
        data['Date_received'], format='%Y%m%d')

    if 'Date' in data.columns.tolist():  # off_train
        #  float类型转换为datetime类型
        data['date'] = pd.to_datetime(data[data['Date'] != 'null']['Date'], format='%Y%m%d')

    # 返回
    return data


def get_label(dataset):
    """打标

    领取优惠券后15天内使用的样本标签为1,否则为0;

    Args:
        dataset: DataFrame类型的数据集off_train,包含属性'User_id','Merchant_id','Coupon_id','Discount_rate',
            'Distance','Date_received','Date'

    Returns:
        打标后的DataFrame类型的数据集.
    """
    # 源数据
    data = dataset.copy()
    # 打标:领券后15天内消费为1,否则为0
    data['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data['date'],
                             data['date_received']))
    # 返回
    return data


def get_week_feature(label_field):
    """根据Date_received得到的一些日期特征

    - 根据date_received列得到领券日是周几,新增一列week存储,并将其one-hot离散为week_0,week_1,week_2,week_3,week_4,week_5,week_6;
    - 根据week列得到领券日是否为休息日,新增一列is_weekend存储;
    - 领取优惠券是一月的第几天

    """
    print('- 获取日期特征')

    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)  # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(
        int)  # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    # 返回的特征数据集
    feature = data.copy()
    print('-- 领取优惠券是星期几')
    feature['week'] = feature['date_received'].map(lambda x: x.weekday())  # 星期几
    print('-- 领券日是否为休息日')
    feature['is_weekend'] = feature['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)  # 判断领券日是否为休息日
    feature = pd.concat([feature, pd.get_dummies(feature['week'], prefix='week')], axis=1)  # one-hot离散星期几

    # 领取优惠券是一月的第几天
    print('-- 领取优惠券是一月的第几天')
    feature['which_day_of_this_month'] = data['Date_received'].map(lambda x: x % 100)

    feature.index = range(len(feature))  # 重置index
    # 返回
    return feature


def get_simple_feature(label_field):
    """获取其他特征
    - 用户领取的所有优惠券数目
	- 用户领取的特定优惠券数目
	- 用户此次之后/前领取的所有优惠券数目
	- 用户此次之后/前领取的特定优惠券数目
	- 用户上/下一次领取的时间间隔
	- 用户上/下一次领取特定优惠卷的时间间隔
	- 用户领取特定商家的优惠券数目
	- 用户领取的不同商家数目
	- 用户当天领取的优惠券数目
	- 用户当天领取的特定优惠券数目
	- 用户领取的所有优惠券种类数目
	- 用户是否在同一天重复领取了特定优惠券
	- 商家被领取的优惠券数目
    - 商家被领取的特定优惠券数目
	- 商家被多少不同用户领取的数目
	- 商家发行的所有优惠券种类数目

    """
    print('- 获取基本特征')

    # 源数据
    data = label_field.copy()
    data['User_id'] = data['User_id'].map(int)
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Merchant_id'] = data['Merchant_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['cnt'] = 1  # 方便特征提取
    # 返回的特征数据集
    feature = data.copy()

    # 用户领券数
    print('-- 用户领券数')
    keys = ['User_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户领取特定优惠券数
    print('-- 用户领取特定优惠券数')
    keys = ['User_id', 'Coupon_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户此次之前 / 后领取的所有优惠券数目
    print('-- 用户此次之前 / 后领取的所有优惠券数目')
    keys = ['User_id']
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    df = pivot.reset_index()
    dic_User_Coupon_cnt = {}
    np_mat = df.values
    for i in range(df.shape[0]):
        dic_User_Coupon_cnt[np_mat[i, 0]] = [np_mat[i, 1], 0]

    np_mat = feature.values
    np_feature = np.zeros([feature.shape[0], 2])
    for i in range(feature.shape[0]):
        d1, d2 = dic_User_Coupon_cnt[np_mat[i, 0]]
        np_feature[i, 0], np_feature[i, 1] = [d1 - 1, d2]
        dic_User_Coupon_cnt[np_mat[i, 0]] = [d1 - 1, d2 + 1]
    feature['simple_User_id_receive_last_cnt'] = np_feature[:, 0]
    feature['simple_User_id_receive_before_cnt'] = np_feature[:, 1]
    feature['simple_User_id_receive_last_cnt'] = feature['simple_User_id_receive_last_cnt'].map(int)
    feature['simple_User_id_receive_before_cnt'] = feature['simple_User_id_receive_before_cnt'].map(int)

    # 用户此次之前 / 后领取的特定优惠券数目
    print('-- 用户此次之前 / 后领取的特定优惠券数目')
    keys = ['User_id', 'Coupon_id']
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    df = pivot.reset_index()
    dic_User_Coupon_cnt = {}
    np_mat = df.values
    for i in range(df.shape[0]):
        dic_User_Coupon_cnt[str(np_mat[i, 0]) + '_' + str(np_mat[i, 1])] = [np_mat[i, 2], 0]

    np_mat = feature.values
    np_feature = np.zeros([feature.shape[0], 2])
    for i in range(feature.shape[0]):
        d1, d2 = dic_User_Coupon_cnt[str(np_mat[i, 0]) + '_' + str(np_mat[i, 2])]
        np_feature[i, 0], np_feature[i, 1] = [d1 - 1, d2]
        dic_User_Coupon_cnt[str(np_mat[i, 0]) + '_' + str(np_mat[i, 2])] = [d1 - 1, d2 + 1]
    feature['simple_User_id_Coupon_id_receive_last_cnt'] = np_feature[:, 0]
    feature['simple_User_id_Coupon_id_receive_before_cnt'] = np_feature[:, 1]
    feature['simple_User_id_Coupon_id_receive_last_cnt'] = feature['simple_User_id_Coupon_id_receive_last_cnt'].map(int)
    feature['simple_User_id_Coupon_id_receive_before_cnt'] = feature['simple_User_id_Coupon_id_receive_before_cnt'].map(
        int)

    # 用户上/下一次领取的时间间隔
    print('-- 用户上/下一次领取的时间间隔')
    keys = ['User_id']
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    df = pivot.reset_index()
    dic_User_Coupon_cnt = {}
    np_mat = df.values
    for i in range(df.shape[0]):
        dic_User_Coupon_cnt[str(np_mat[i, 0])] = -1

    np_mat = feature.values
    np_feature = np.full([feature.shape[0], 2], -1)
    for i in range(feature.shape[0]):
        d = dic_User_Coupon_cnt[str(np_mat[i, 0])]
        if (d >= 0):
            t = (datetime.strptime(str(np_mat[i, 5]), '%Y%m%d') - datetime.strptime(str(np_mat[d, 5]), '%Y%m%d')).days
            np_feature[i, 0] = t
            np_feature[d, 1] = t
        dic_User_Coupon_cnt[str(np_mat[i, 0])] = i
    feature['simple_User_id_receive_last_gap'] = np_feature[:, 0]
    feature['simple_User_id_receive_before_gap'] = np_feature[:, 1]
    feature['simple_User_id_receive_last_cnt'] = feature['simple_User_id_receive_last_gap'].map(int)
    feature['simple_User_id_receive_before_gap'] = feature['simple_User_id_receive_before_gap'].map(
        int)

    # 用户上/下一次领取特定优惠卷的时间间隔
    print('-- 用户上/下一次领取特定优惠卷的时间间隔')
    keys = ['User_id', 'Coupon_id']
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    df = pivot.reset_index()
    dic_User_Coupon_cnt = {}
    np_mat = df.values
    for i in range(df.shape[0]):
        dic_User_Coupon_cnt[str(np_mat[i, 0]) + '_' + str(np_mat[i, 1])] = -1

    np_mat = feature.values
    np_feature = np.full([feature.shape[0], 2], -1)
    for i in range(feature.shape[0]):
        d = dic_User_Coupon_cnt[str(np_mat[i, 0]) + '_' + str(np_mat[i, 2])]
        if (d >= 0):
            t = (datetime.strptime(str(np_mat[i, 5]), '%Y%m%d') - datetime.strptime(str(np_mat[d, 5]), '%Y%m%d')).days
            np_feature[i, 0] = t
            np_feature[d, 1] = t
        dic_User_Coupon_cnt[str(np_mat[i, 0]) + '_' + str(np_mat[i, 2])] = i
    feature['simple_User_id_Coupon_id_receive_last_gap'] = np_feature[:, 0]
    feature['simple_User_id_Coupon_id_receive_before_gap'] = np_feature[:, 1]
    feature['simple_User_id_Coupon_id_receive_last_gap'] = feature['simple_User_id_Coupon_id_receive_last_gap'].map(int)
    feature['simple_User_id_Coupon_id_receive_before_gap'] = feature['simple_User_id_Coupon_id_receive_before_gap'].map(
        int)

    # 用户领取特定商家的优惠券数目
    print('-- 用户领取特定商家的优惠券数目')
    keys = ['User_id', 'Merchant_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户领取的不同商家数目
    print('-- 用户领取的不同商家数目')
    keys = ['User_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'Merchant_id': prefixs + 'receive_differ_Merchant_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户当天领券数
    print('-- 用户当天领券数')
    keys = ['User_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户当天领取特定优惠券数
    print('-- 用户当天领取特定优惠券数')
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户领取的所有优惠券种类数目
    print('-- 用户领取的所有优惠券种类数目')
    keys = ['User_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'Coupon_id': prefixs + 'receive_differ_Coupon_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户是否在同一天重复领取了特定优惠券
    print('-- 用户是否在同一天重复领取了特定优惠券')
    keys = ['User_id', 'Coupon_id', 'Date_received']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt',
                           aggfunc=lambda x: 1 if len(x) > 1 else 0)  # 以keys为键,'cnt'为值,判断领取次数是否大于1
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'repeat_receive'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 商家被领取的优惠券数目
    print('-- 商家被领取的优惠券数目')
    keys = ['Merchant_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 商家被领取的特定优惠券数目
    print('-- 商家被领取的特定优惠券数目')
    keys = ['Merchant_id', 'Coupon_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 商家被多少不同用户领取的数目
    print('-- 商家被多少不同用户领取的数目')
    keys = ['Merchant_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'User_id': prefixs + 'receive_differ_User_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 商家发行的所有优惠券种类数目
    print('-- 商家发行的所有优惠券种类数目')
    keys = ['Merchant_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'Coupon_id': prefixs + 'differ_Coupon_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    feature.fillna(0, downcast='infer', inplace=True)

    # 删除辅助提特征的'cnt'
    feature.drop(['cnt'], axis=1, inplace=True)

    # 返回
    return feature


def get_label_field_feature(label_field):
    week_feat = get_week_feature(label_field)
    simple_feat = get_simple_feature(label_field)

    # 构造数据集
    share_characters = list(
        set(simple_feat.columns.tolist()) & set(week_feat.columns.tolist()))  # 共有属性,包括id和一些基础特征,为每个特征块的交集
    label_field = pd.concat([week_feat, simple_feat.drop(share_characters, axis=1)], axis=1)

    return label_field


def get_middle_field_feature(label_field, middle_field):
    """
    写入提取中间特征的代码，提取的特征按需要的主键连接到label_field，返回label_field(这里作为示例没有补全)
    """
    return label_field


def get_history_field_merchant_feature(label_field, history_field):
    """历史区间的商家相关的特征

    - 历史上商家优惠券被领取次数
	- 历史上商家优惠券被领取后不核销次数
	- 历史上商家优惠券被领取后核销次数
	- 历史上商家优惠券被领取后核销率
	- 历史上商家优惠券核销的平均/最小/最大消费折率
	- 历史上商家提供的不同优惠卷数目
	- 历史上领取商家优惠券的不同用户数量
	- 历史上核销商家优惠券的不同用户数量，及其占领取不同的用户比重
	- 历史上商家优惠券平均每个用户核销多少张
	- 历史上商家被核销过的不同优惠券数量
	- 历史上商家平均每种优惠券核销多少张
	- 历史上商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
	历史上商家被核销优惠券的平均时间率
	- 历史上商家被核销优惠券中的平均/最小/最大用户-商家距离

    """
    print('- 历史区间的商家相关的特征')

    # 源数据
    data = history_field.copy()
    # 将'Merchant_id'列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Merchant_id'] = data['Merchant_id'].map(int)
    # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(int)
    # 方便特征提取
    data['cnt'] = 1

    keys = ['Merchant_id']
    # 特征名前缀,由history_field和主键组成
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    # 返回的特征数据集
    feature = label_field[keys].drop_duplicates(keep='first')

    # 历史上商家优惠券被领取次数
    print('-- 历史上商家优惠券被领取次数')
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上商家优惠券被领取后不核销次数
    print('-- 历史上商家优惠券被领取后不核销次数')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == 'null')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_and_unconsume_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上商家优惠券被领取后核销次数
    print('-- 历史上商家优惠券被领取后核销次数')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户对该优惠券的核销率
    print('-- 历史上用户对该优惠券的核销率')
    feature['rate'] = feature.apply(lambda x:
                                    x[prefixs + 'receive_and_consume_cnt']
                                    / x[prefixs + 'receive_cnt']
                                    if x[prefixs + 'receive_cnt'] > 0
                                    else 0, axis=1)
    feature.rename(columns={'rate': prefixs + 'receive_and_consume_rate'}, inplace=True)

    # 历史上商家优惠券核销的平均消费折率
    print('-- 历史上商家优惠券核销的平均消费折率')
    pivot = pd.pivot_table(data, index=keys, values='discount_rate', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={
        'discount_rate': prefixs + 'receive_and_consume_mean_rate'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上商家优惠券核销的最小消费折率
    print('-- 历史上商家优惠券核销的最小消费折率')
    pivot = pd.pivot_table(data, index=keys, values='discount_rate', aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={
        'discount_rate': prefixs + 'receive_and_consume_min_rate'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上商家优惠券核销的最大消费折率
    print('-- 历史上商家优惠券核销的最大消费折率')
    pivot = pd.pivot_table(data, index=keys, values='discount_rate', aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={
        'discount_rate': prefixs + 'receive_and_consume_max_rate'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上商家提供的不同优惠卷数目
    print('-- 历史上商家提供的不同优惠卷数目')
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'Coupon_id': prefixs + 'differ_Coupon_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上领取商家优惠券的不同用户数量
    print('-- 历史上领取商家优惠券的不同用户数量')
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'User_id': prefixs + 'differ_User_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上核销商家优惠券的不同用户数量
    print('-- 历史上核销商家优惠券的不同用户数量')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='User_id',
                           aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'User_id': prefixs + 'receive_and_consume_differ_User_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上核销商家优惠券的不同用户数量占领取不同的用户比重
    print('-- 历史上核销商家优惠券的不同用户数量占领取不同的用户比重')
    differ_User = len(set(data['User_id']))
    feature[prefixs + 'receive_and_consume_differ_User_differ_User_rate'] = list(
        map(lambda x: x / differ_User if differ_User > 0 else 0,
            feature[prefixs + 'receive_and_consume_differ_User_cnt']))

    # 历史上商家优惠券平均每个用户核销多少张
    print('-- 历史上商家优惠券平均每个用户核销多少张')
    feature[prefixs + 'each_User_consume_cnt'] = list(
        map(lambda x, y: x / y if y > 0 else 0, feature[prefixs + 'receive_cnt'], feature[prefixs + 'differ_User_cnt']))

    # 历史上商家被核销过的不同优惠券数量
    print('-- 历史上商家被核销过的不同优惠券数量')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='Coupon_id',
                           aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'Coupon_id': prefixs + 'receive_and_consume_differ_Coupon_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上商家平均每种优惠券核销多少张
    print('-- 历史上商家平均每种优惠券核销多少张')
    feature[prefixs + 'each_Coupon_consume_cnt'] = list(
        map(lambda x, y: x / y if y > 0 else 0, feature[prefixs + 'receive_cnt'],
            feature[prefixs + 'differ_Coupon_cnt']))

    # 历史上商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
    print('-- 历史上商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重')
    differ_Coupon = len(set(data['Coupon_id']))
    feature[prefixs + 'receive_and_consume_differ_Coupon_receive_differ_Coupon_rate'] = list(
        map(lambda x: x / differ_Coupon if differ_Coupon > 0 else 0,
            feature[prefixs + 'receive_and_consume_differ_Coupon_cnt']))

    # 历史上商家被核销优惠券的平均时间率

    # 历史上商家被核销优惠券中的平均用户 - 商家距离
    print('-- 历史上商家被核销优惠券中的平均用户 - 商家距离')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='Distance',
                           aggfunc=lambda x: sum([i for i in x if i >= 0]) / len(x) if len(x) > 0 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={
        'Distance': prefixs + 'receive_and_consume_Distance_mean'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上商家被核销优惠券中的最小用户 - 商家距离
    print('-- 历史上商家被核销优惠券中的最小用户 - 商家距离')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='Distance',
                           aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={
        'Distance': prefixs + 'receive_and_consume_Distance_min'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上商家被核销优惠券中的最大用户 - 商家距离
    print('-- 历史上商家被核销优惠券中的最大用户 - 商家距离')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='Distance',
                           aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={
        'Distance': prefixs + 'receive_and_consume_Distance_max'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    feature.fillna(0, downcast='infer', inplace=True)

    # 返回
    return feature


def get_history_field_coupon_feature(label_field, history_field):
    """历史区间的优惠券相关的特征

    - 历史上该优惠卷被领取次数
	- 历史上该优惠卷被消费次数
	- 历史上该优惠卷未被消费次数
	- 历史上该优惠卷被消费率
	- 历史上该优惠卷15天内核销时间率(1-(消费时间-领取时间)/15，如果未核销则为0，即越早核销值越大)
	- 历史上该优惠卷15天内被核销的平均时间间隔
	- 历史上满减优惠券最低消费的中位数

    """
    print('- 历史区间的优惠券相关的特征')

    # 源数据
    data = history_field.copy()
    # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Coupon_id'] = data['Coupon_id'].map(int)
    # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(int)
    # 方便特征提取
    data['cnt'] = 1

    keys = ['Coupon_id']
    # 特征名前缀,由history_field和主键组成
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    # 返回的特征数据集
    feature = label_field[keys].drop_duplicates(keep='first')

    # 历史上该优惠卷被领取次数
    print('-- 历史上该优惠卷被领取次数')
    pivot = pd.pivot_table(data, index=['Coupon_id'], values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(
        columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上该优惠卷被消费次数
    print('-- 历史上该优惠卷被消费次数')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(
        columns={'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上该优惠卷未被消费次数
    print('-- 历史上该优惠卷未被消费次数')
    feature['cnt'] = feature.apply(lambda x:
                                   x['history_field_Coupon_id_receive_cnt']
                                   - x['history_field_Coupon_id_receive_and_consume_cnt'], axis=1)
    feature.rename(columns={'cnt': prefixs + 'receive_and_unconsume_cnt'}, inplace=True)

    # 历史上该优惠卷被消费率
    print('-- 历史上该优惠卷被消费率')
    feature['rate'] = feature.apply(lambda x:
                                    x['history_field_Coupon_id_receive_and_consume_cnt']
                                    / x['history_field_Coupon_id_receive_cnt']
                                    if x['history_field_Coupon_id_receive_cnt'] > 0
                                    else 0, axis=1)
    feature.rename(columns={'rate': prefixs + 'receive_and_consume_rate'}, inplace=True)

    # 历史上该优惠卷15天内核销时间率
    print('-- 历史上该优惠卷15天内核销时间率')
    feature['rate'] = data.apply(lambda x:
                                 1 - (x['date'] - x['date_received']).total_seconds() / (60 * 60 * 24) / 15
                                 if x['label'] == 1
                                 else 0, axis=1)
    feature.rename(columns={'rate': prefixs + 'receive_and_consume_time_rate'}, inplace=True)

    # 历史上该优惠卷15天内被核销的平均时间间隔
    print('-- 历史上该优惠卷15天内被核销的平均时间间隔')
    tmp = data[data['label'] == 1]
    # 核销与领券的时间间隔,以天为单位
    tmp['gap'] = (tmp['date'] - tmp['date_received']
                  ).map(lambda x: x.total_seconds() / (60 * 60 * 24))
    # 以keys为键,'gap'为值,使用np.mean统计平均时间间隔
    pivot = pd.pivot_table(tmp, index=keys, values='gap', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(
        columns={'gap': prefixs + 'consumed_mean_time_gap_15'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature[prefixs + 'consumed_mean_time_gap_15'].fillna(-1, downcast='infer', inplace=True)

    # 满减优惠券最低消费的中位数
    print('-- 历史上满减优惠券最低消费的中位数')
    # 先筛选出is_manjian为1即满减券的样本,以keys为键,'min_cost_of_manjian'为值,使用np.median统计中位数
    pivot = pd.pivot_table(data[data['is_manjian'] == 1], index=keys, values='min_cost_of_manjian',
                           aggfunc=np.median)
    pivot = pd.DataFrame(pivot).rename(
        columns={'min_cost_of_manjian': prefixs + 'median_of_min_cost_of_manjian'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    feature.fillna(0, downcast='infer', inplace=True)

    # 返回
    return feature


def get_history_field_merchant_coupon_feature(label_field, history_field):
    """历史区间的商家-优惠券相关的特征

    - 历史上商家被领取的特定优惠券数目

    """
    print('- 历史区间的商家-优惠券相关的特征')

    # 源数据
    data = history_field.copy()
    # 将'Merchant_id'列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Merchant_id'] = data['Merchant_id'].map(int)
    # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Coupon_id'] = data['Coupon_id'].map(int)
    # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(int)
    # 方便特征提取
    data['cnt'] = 1

    keys = ['Merchant_id', 'Coupon_id']
    # 特征名前缀,由history_field和主键组成
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    # 返回的特征数据集
    feature = label_field[keys].drop_duplicates(keep='first')

    # 历史上商家被领取的特定优惠券数目
    print('-- 历史上商家被领取的特定优惠券数目')
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    feature.fillna(0, downcast='infer', inplace=True)

    # 返回
    return feature


def get_history_field_user_feature(label_field, history_field):
    """历史区间的用户相关的特征

    - 历史上用户领取优惠券次数
	- 历史上用户获得优惠券但没有消费的次数
	- 历史上用户获得优惠券并核销次数
	- 历史上用户领取优惠券后进行核销率
    - 历史上用户核销优惠券的平均/最低/最高消费折率
    - 历史上用户领取多少个不同的商家
    - 历史上用户领取不同的商家数量占所有不同商家的比重
	- 历史上用户核销过优惠券的不同商家数量，及其占所有不同商家的比重
	- 历史上用户核销过优惠券的不同商家数量占领取过的不同商家的数量的比重
	- 历史上用户核销过的不同优惠券数量，及其占所有不同优惠券的比重
	- 历史上用户对领卷商家的15天内的核销数
	- 历史上用户平均核销每个商家多少张优惠券
	- 历史上用户核销优惠券中的平均/最大/最小用户-商家距离

    """
    print('- 历史区间的用户相关的特征')

    # 源数据
    data = history_field.copy()
    # 将User_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['User_id'] = data['User_id'].map(int)
    # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(int)
    # 方便特征提取
    data['cnt'] = 1

    keys = ['User_id']
    # 特征名前缀,由history_field和主键组成
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    # 返回的特征数据集
    feature = label_field[keys].drop_duplicates(keep='first')

    # 历史上用户领取优惠券次数
    print('-- 历史上用户领取优惠券次数')
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户获得优惠券但没有消费的次数
    print('-- 历史上用户获得优惠券但没有消费的次数')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == 'null')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_and_unconsume_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户获得优惠券并核销次数
    print('-- 历史上用户获得优惠券并核销次数')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户领取优惠券后进行核销率
    print('-- 历史上用户领取优惠券后进行核销率')
    feature[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y: x / y if y > 0 else 0,
                                                             feature[prefixs + 'receive_and_consume_cnt'],
                                                             feature[prefixs + 'receive_cnt']))

    # 历史上用户核销优惠券的平均消费折率
    print('-- 历史上用户核销优惠券的平均消费折率')
    pivot = pd.pivot_table(data, index=keys, values='discount_rate', aggfunc=np.mean)
    pivot = pd.DataFrame(pivot).rename(columns={
        'discount_rate': prefixs + 'receive_and_consume_mean_rate'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户核销优惠券的最低消费折率
    print('-- 历史上用户核销优惠券的最低消费折率')
    pivot = pd.pivot_table(data, index=keys, values='discount_rate', aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={
        'discount_rate': prefixs + 'receive_and_consume_min_rate'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户核销优惠券的最高消费折率
    print('-- 历史上用户核销优惠券的最高消费折率')
    pivot = pd.pivot_table(data, index=keys, values='discount_rate', aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={
        'discount_rate': prefixs + 'receive_and_consume_max_rate'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户领取多少个不同的商家
    print('-- 历史上用户领取多少个不同的商家')
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'Merchant_id': prefixs + 'receive_differ_Merchant_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')
    feature[prefixs + 'receive_differ_Merchant_cnt'].fillna(0, downcast='infer', inplace=True)

    # 历史上用户核销过优惠券的不同商家数量
    print('-- 历史上用户核销过优惠券的不同商家数量')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='Merchant_id',
                           aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'Merchant_id': prefixs + 'receive_and_consume_differ_Merchant_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户领取不同的商家数量占所有不同商家的比重
    print('-- 历史上用户领取不同的商家数量占所有不同商家的比重')
    differ_Merchant_cnt = len(set(data['Merchant_id']))
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)) / differ_Merchant_cnt)
    pivot = pd.DataFrame(pivot).rename(columns={
        'Merchant_id': prefixs + 'receive_and_consume_differ_Merchant_alldiffer_rate'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户核销过优惠券的不同商家数量占所有不同商家的比重
    print('-- 历史上用户核销过优惠券的不同商家数量占所有不同商家的比重')
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id',
                           aggfunc=lambda x: len(set(x)) / differ_Merchant_cnt)
    pivot = pd.DataFrame(pivot).rename(columns={
        'Merchant_id': prefixs + 'receive_and_consume_differ_Merchant_alldiffer_rate'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户核销过优惠券的不同商家数量占领取过的不同商家的数量的比重
    print('-- 历史上用户核销过优惠券的不同商家数量占领取过的不同商家的数量的比重')
    feature[prefixs + 'receive_and_consume_differ_Merchant_receive_differ_rate'] = list(
        map(lambda x, y: x / y if y > 0 else 0,
            feature[prefixs + 'receive_and_consume_differ_Merchant_cnt'],
            feature[prefixs + 'receive_differ_Merchant_cnt']))

    # 历史上用户领取多少个不同的优惠卷
    print('-- 历史上用户领取多少个不同的优惠卷')
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'Coupon_id': prefixs + 'receive_and_consume_differ_Coupon_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户领取的不同的优惠卷数量占所有不同优惠卷的比重
    print('-- 历史上用户领取的不同的优惠卷数量占所有不同优惠卷的比重')
    differ_Coupon_cnt = len(set(data['Coupon_id']))
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id',
                           aggfunc=lambda x: len(set(x)) / differ_Coupon_cnt)
    pivot = pd.DataFrame(pivot).rename(columns={
        'Coupon_id': prefixs + 'receive_and_consume_differ_Coupon_alldiffer_rate'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户对领卷商家的15天内的核销数
    print('-- 历史上用户对领卷商家的15天内的核销数')
    pivot = pd.pivot_table(data[data['label'] == 1], index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot = pd.DataFrame(pivot).rename(columns={
        'Merchant_id': prefixs + 'receive_and_consume_differ_Merchant_cnt_15'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户平均核销每个商家多少张优惠券
    print('-- 历史上用户平均核销每个商家多少张优惠券')
    feature[prefixs + 'receive_and_consume_differ_Merchant_cnt_mean'] = list(map(lambda x, y: x / y if y > 0 else 0,
                                                                                 feature[
                                                                                     prefixs + 'receive_and_consume_cnt'],
                                                                                 feature[
                                                                                     prefixs + 'receive_differ_Merchant_cnt']))

    # 历史上用户核销优惠券中的平均用户 - 商家距离
    print('-- 历史上用户核销优惠券中的平均用户 - 商家距离')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='Distance',
                           aggfunc=lambda x: sum([i for i in x if i >= 0]) / len(x) if len(x) > 0 else 0)
    pivot = pd.DataFrame(pivot).rename(columns={
        'Distance': prefixs + 'receive_and_consume_Distance_mean'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户核销优惠券中的最大用户 - 商家距离
    print('-- 历史上用户核销优惠券中的最大用户 - 商家距离')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='Distance',
                           aggfunc=np.max)
    pivot = pd.DataFrame(pivot).rename(columns={
        'Distance': prefixs + 'receive_and_consume_Distance_max'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户核销优惠券中的最小用户 - 商家距离
    print('-- 历史上用户核销优惠券中的最小用户 - 商家距离')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='Distance',
                           aggfunc=np.min)
    pivot = pd.DataFrame(pivot).rename(columns={
        'Distance': prefixs + 'receive_and_consume_Distance_min'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    feature.fillna(0, downcast='infer', inplace=True)

    # 返回
    return feature


def get_history_field_user_coupon_feature(label_field, history_field):
    """历史区间的用户-优惠券相关的特征

    - 历史上用户领取特定优券次惠数
	- 历史上用户消费特定优券次惠数
	- 历史上用户对特定优惠券的核销率
	历史上用户满0~50/50~200/200~500 减的优惠券核销率
	历史上用户核销满0~50/50~200/200~500减的优惠券占所有核销优惠券的比重

    """
    print('- 历史区间的用户-优惠券相关的特征')

    # 源数据
    data = history_field.copy()
    # 将User_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['User_id'] = data['User_id'].map(int)
    # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Coupon_id'] = data['Coupon_id'].map(int)
    # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(int)
    # 方便特征提取
    data['cnt'] = 1

    keys = ['User_id', 'Coupon_id']
    # 特征名前缀,由history_field和主键组成
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    # 返回的特征数据集
    feature = label_field[keys].drop_duplicates(keep='first')

    # 历史上用户领取特定优券次惠数
    print('-- 历史上用户领取特定优券次惠数')
    pivot = pd.pivot_table(data, index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户消费特定优券次惠数
    print('-- 历史上用户消费特定优券次惠数')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户对特定优惠券的核销率
    print('-- 历史上用户对特定优惠券的核销率')
    feature[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y: x / y if y > 0 else 0,
                                                             feature[prefixs + 'receive_and_consume_cnt'],
                                                             feature[prefixs + 'receive_cnt']))

    # 历史上用户满0~50 / 50~200 / 200~500 减的优惠券核销率

    # 历史上用户核销满0~50 / 50~200 / 200~500 减的优惠券占所有核销优惠券的比重

    feature.fillna(0, downcast='infer', inplace=True)

    # 返回
    return feature


def get_history_field_user_merchant_feature(label_field, history_field):
    """历史区间的用户-商家相关的特征

    - 历史上用户领取商家的优惠券次数
	- 历史上用户领取商家的优惠券后不核销次数
	- 历史上用户领取商家的优惠券后核销次数
	- 历史上用户领取商家的优惠券后核销率

    """
    print('- 历史区间的用户-商家相关的特征')

    # 源数据
    data = history_field.copy()
    # 将User_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['User_id'] = data['User_id'].map(int)
    # 将Merchant_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Merchant_id'] = data['Merchant_id'].map(int)
    # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(int)
    # 方便特征提取
    data['cnt'] = 1

    keys = ['User_id', 'Merchant_id']
    # 特征名前缀,由history_field和主键组成
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    # 返回的特征数据集
    feature = label_field[keys].drop_duplicates(keep='first')

    # 历史上用户领取商家的优惠券次数
    print('-- 历史上用户领取商家的优惠券次数')
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户领取商家的优惠券后不核销次数
    print('-- 历史上用户领取商家的优惠券后不核销次数')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) == 'null')], index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_and_unconsume_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户领取商家的优惠券后核销次数
    print('-- 历史上用户领取商家的优惠券后核销次数')
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 历史上用户领取商家的优惠券后核销率
    print('-- 历史上用户领取商家的优惠券后核销率')
    feature[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y: x / y if y > 0 else 0,
                                                             feature[prefixs + 'receive_and_consume_cnt'],
                                                             feature[prefixs + 'receive_cnt']))

    feature.fillna(0, downcast='infer', inplace=True)

    # 返回
    return feature


def get_history_field_user_merchant_coupon_feature(history_feat):
    """用户-商家-优惠卷综合特征

    - 历史上用户对每个商家的不核销次数占用户总的不核销次数的比重
	- 历史上用户对每个商家的优惠券核销次数占用户总的核销次数的比重
	- 历史上用户对每个商家的不核销次数占商家总的不核销次数的比重
	- 历史上用户对每个商家的优惠券核销次数占商家总的核销次数的比重

    """
    print('- 用户-商家-优惠卷综合特征')

    # 历史上用户对每个商家的不核销次数占用户总的不核销次数的比重
    print('-- 历史上用户对每个商家的不核销次数占用户总的不核销次数的比重')
    keys = ['User_id', 'Merchant_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    history_feat[prefixs + 'receive_and_unconsume_all_receive_and_unconsume_rate'] = list(
        map(lambda x, y: x / y if y > 0 else 0,
            history_feat[prefixs + 'receive_and_unconsume_cnt'],
            history_feat['history_field_User_id_receive_and_unconsume_cnt']))

    # 历史上用户对每个商家的优惠券核销次数占用户总的核销次数的比重
    print('-- 历史上用户对每个商家的优惠券核销次数占用户总的核销次数的比重')
    keys = ['User_id', 'Merchant_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    history_feat[prefixs + 'receive_and_consume_all_receive_and_consume_rate'] = list(
        map(lambda x, y: x / y if y > 0 else 0,
            history_feat[prefixs + 'receive_and_consume_cnt'],
            history_feat['history_field_User_id_receive_and_consume_cnt']))

    # 历史上用户对每个商家的不核销次数占商家总的不核销次数的比重
    print('-- 历史上用户对每个商家的不核销次数占商家总的不核销次数的比重')
    keys = ['User_id', 'Merchant_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    history_feat[prefixs + 'receive_and_consume_Merchant_all_receive_and_unconsume_rate'] = list(
        map(lambda x, y: x / y if y > 0 else 0,
            history_feat[prefixs + 'receive_and_unconsume_cnt'],
            history_feat['history_field_Merchant_id_receive_and_unconsume_cnt']))

    # 历史上用户对每个商家的优惠券核销次数占商家总的核销次数的比重
    print('-- 历史上用户对每个商家的优惠券核销次数占商家总的核销次数的比重')
    keys = ['User_id', 'Merchant_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    history_feat[prefixs + 'receive_and_consume_Merchant_all_receive_and_consume_rate'] = list(
        map(lambda x, y: x / y if y > 0 else 0,
            history_feat[prefixs + 'receive_and_consume_cnt'],
            history_feat['history_field_Merchant_id_receive_and_consume_cnt']))

    # 历史上商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重
    print('-- 历史上商家被核销过的不同优惠券数量占所有领取过的不同优惠券数量的比重')

    history_feat.fillna(0, downcast='infer', inplace=True)

    return history_feat


def get_history_field_feature(label_field, history_field):
    # 商家特征
    m_feat = get_history_field_merchant_feature(label_field, history_field)

    # 优惠卷特征
    c_feat = get_history_field_coupon_feature(label_field, history_field)

    # 用户特征
    u_feat = get_history_field_user_feature(label_field, history_field)

    # 商家-优惠卷特征
    mc_feat = get_history_field_merchant_coupon_feature(label_field, history_field)

    # 用户-优惠卷特征
    uc_feat = get_history_field_user_coupon_feature(label_field, history_field)

    # 用户-商家特征
    um_feat = get_history_field_user_merchant_feature(label_field, history_field)

    # 添加特征
    history_feat = label_field.copy()

    history_feat = pd.merge(history_feat, m_feat, on=['Merchant_id'], how='left')

    history_feat = pd.merge(history_feat, c_feat, on=['Coupon_id'], how='left')

    history_feat = pd.merge(history_feat, u_feat, on=['User_id'], how='left')

    history_feat = pd.merge(history_feat, mc_feat, on=['Merchant_id', 'Coupon_id'], how='left')

    history_feat = pd.merge(history_feat, uc_feat, on=['User_id', 'Coupon_id'], how='left')

    history_feat = pd.merge(history_feat, um_feat, on=['User_id', 'Merchant_id'], how='left')

    # 用户-商家-优惠卷综合特征
    history_feat = get_history_field_user_merchant_coupon_feature(history_feat)

    # 返回
    return history_feat


def get_dataset(history_field, middle_field, label_field):
    """构造数据集

    """
    # 特征工程

    # 标签区间特征
    label_feat = get_label_field_feature(label_field)
    # 中间区间特征
    middle_feat = get_middle_field_feature(label_field, middle_field)
    # 历史区间特征
    history_feat = get_history_field_feature(label_field, history_field)
    # 构造数据集
    # 共有属性,包括id和一些基础特征,为每个特征块的交集
    share_characters = list(set(history_feat.columns.tolist()) &
                            set(middle_feat.columns.tolist()) &
                            set(label_feat.columns.tolist()))
    # 这里使用concat连接而不用merge,因为几个特征块的样本顺序一致,index一致,但需要注意在连接两个特征块时要删去其中一个特征块的共有属性
    dataset = pd.concat(
        [history_feat, middle_feat.drop(share_characters, axis=1)], axis=1, join_axes=[history_feat.index])
    dataset = pd.concat(
        [dataset, label_feat.drop(share_characters, axis=1)], axis=1, join_axes=[dataset.index])

    # 删除无用属性并将label置于最后一列
    if 'Date' in dataset.columns.tolist():  # 表示训练集和验证集
        dataset.drop(['Merchant_id', 'Discount_rate', 'Date', 'date_received', 'date'], axis=1, inplace=True)
        label = dataset['label'].tolist()
        dataset.drop(['label'], axis=1, inplace=True)
        dataset['label'] = label
    else:  # 表示测试集
        dataset.drop(['Merchant_id', 'Discount_rate', 'date_received'], axis=1, inplace=True)
    # 修正数据类型
    dataset['User_id'] = dataset['User_id'].map(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].map(int)
    dataset['Date_received'] = dataset['Date_received'].map(int)
    dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(int)
    # 去重
    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.index = range(len(dataset))
    # 返回
    return dataset


if __name__ == '__main__':
    start = datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))

    # 源数据
    off_train = pd.read_csv(r'./input_files/ccf_offline_stage1_train.csv')
    off_test = pd.read_csv(r'./input_files/ccf_offline_stage1_test_revised.csv')
    # 预处理
    off_train = prepare(off_train)
    off_test = prepare(off_test)
    # 打标
    off_train = get_label(off_train)

    # 划分区间
    # 训练集历史区间、中间区间、标签区间
    train_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501)
    train_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/5/1', periods=15))]  # [20160501,20160516)
    train_label_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]  # [20160516,20160616)
    # 验证集历史区间、中间区间、标签区间
    validate_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]  # [20160116,20160316)
    validate_middle_field = off_train[
        off_train['date'].isin(pd.date_range('2016/3/16', periods=15))]  # [20160316,20160331)
    validate_label_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/3/31', periods=31))]  # [20160331,20160501)
    # 测试集历史区间、中间区间、标签区间
    test_history_field = off_train[
        off_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]  # [20160417,20160616)
    test_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/6/16', periods=15))]  # [20160616,20160701)
    test_label_field = off_test.copy()  # [20160701,20160801)

    # 构造训练集、验证集、测试集
    print('\n构造训练集')
    train = get_dataset(train_history_field, train_middle_field, train_label_field)
    print('\n构造验证集')
    validate = get_dataset(validate_history_field, validate_middle_field, validate_label_field)
    print('\n构造测试集')
    test = get_dataset(test_history_field, test_middle_field, test_label_field)

    # 保存数据集
    train.to_csv("./prepared_dataset/train.csv")
    validate.to_csv("./prepared_dataset/validate.csv")
    test.to_csv("./prepared_dataset/test.csv")

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('time costed is: %d min' % (int((datetime.now() - start).seconds) / 60))