# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:37:21 2018

@author: FNo0
"""

import pandas as pd
import xgboost as xgb
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
    # data['Distance'].fillna(-1, inplace=True)
    data.ix[data['Distance'] == 'null', 'Distance'] = -1
    # 判断是否是空距离
    data['null_distance'] = data['Distance'].map(lambda x: 1 if x == -1 else 0)

    print("时间处理...")
    # 时间处理
    #  float类型转换为datetime类型
    data['date_received'] = pd.to_datetime(
        data['Date_received'], format='%Y%m%d')

    if 'Date' in data.columns.tolist():  # off_train
        #  float类型转换为datetime类型
        data['date'] = pd.to_datetime(data[data['Date']!='null']['Date'], format='%Y%m%d')

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


def get_simple_feature(label_field):
    """

    Args:

    Returns:

    """
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)  # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(int)  
	# 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['cnt'] = 1  # 方便特征提取
    # 返回的特征数据集
    feature = data.copy()

    # 用户领券数
    keys = ['User_id']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 用户领取特定优惠券数
    keys = ['User_id', 'Coupon_id']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 用户领取特定商家的优惠券数目
    keys = ['User_id', 'Merchant_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 用户当天领券数
    keys = ['User_id', 'Date_received']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 用户当天领取特定优惠券数
    keys = ['User_id', 'Coupon_id', 'Date_received']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 用户是否在同一天重复领取了特定优惠券
    keys = ['User_id', 'Coupon_id', 'Date_received']  # 主键
    prefixs = 'simple_' + '_'.join(keys) + '_'  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(data, index=keys, values='cnt',
                           aggfunc=lambda x: 1 if len(x) > 1 else 0)  # 以keys为键,'cnt'为值,判断领取次数是否大于1
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'repeat_receive'}).reset_index()  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how='left')  # 将id列与特征列左连

    # 商家被领取的优惠券数目
    keys = ['Merchant_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 商家被领取的特定优惠券数目
    keys = ['Merchant_id','Coupon_id']
    prefixs = 'simple_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(columns={
        'cnt': prefixs + 'receive_cnt'}).reset_index()
    feature = pd.merge(feature, pivot, on=keys, how='left')

    # 删除辅助提特征的'cnt'
    feature.drop(['cnt'], axis=1, inplace=True)

    # 返回
    return feature


def get_week_feature(label_field):
    """根据Date_received得到的一些日期特征

    根据date_received列得到领券日是周几,新增一列week存储,并将其one-hot离散为week_0,week_1,week_2,week_3,week_4,week_5,week_6;
    根据week列得到领券日是否为休息日,新增一列is_weekend存储;

    Args:

    Returns:

    """
    # 源数据
    data = label_field.copy()
    data['Coupon_id'] = data['Coupon_id'].map(int)  # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(
        int)  # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    # 返回的特征数据集
    feature = data.copy()
    feature['week'] = feature['date_received'].map(lambda x: x.weekday())  # 星期几
    feature['is_weekend'] = feature['week'].map(lambda x: 1 if x == 5 or x == 6 else 0)  # 判断领券日是否为休息日
    feature = pd.concat([feature, pd.get_dummies(feature['week'], prefix='week')], axis=1)  # one-hot离散星期几

    # 领取优惠券是一月的第几天
    feature['which_day_of_this_month']=data['Date_received'].map(lambda x: x%100)

    feature.index = range(len(feature))  # 重置index
    # 返回
    return feature


def get_label_field_feature(label_field):

    week_feat = get_week_feature(label_field)  # 日期特征
    simple_feat = get_simple_feature(label_field)  # 示例简单特征

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


def get_history_field_user_coupon_feature(label_field, history_field):
    """历史区间的用户-优惠券特征

    1.'history_field_User_id_Coupon_id_receive_cnt':
        历史上用户领取该优惠券次数;
    2.'history_field_User_id_Coupon_id_receive_and_consume_cnt':
        历史上用户消费该优惠券次数;
    3.'history_field_User_id_Coupon_id_receive_and_unconsume_cnt':
        历史上用户未消费该优惠券次数;
    4.'history_field_User_id_Coupon_id_receive_and_consume_rate':
        历史上用户对该优惠券的核销率;
    5.'history_field_Coupon_id_receive_cnt':
        历史上该优惠卷被领取次数;
    6.'history_field_Coupon_id_receive_and_consume_cnt':
        历史上该优惠卷被消费次数;
    7.'history_field_Coupon_id_receive_and_unconsume_cnt':
        历史上该优惠卷未被消费次数;
    8.'history_field_Coupon_id_receive_and_consume_rate':
        历史上该优惠卷被消费率;

    Args:
        label_field: 标签区间, DataFrame类型的数据集;
        history_field: 历史区间, DataFrame类型的数据集;

    Returns:
        feature: 提取完特征后的DataFrame类型的数据集.
    """
    # 源数据
    data = history_field.copy()
    # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Coupon_id'] = data['Coupon_id'].map(int)
    # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data['Date_received'] = data['Date_received'].map(int)
    # 方便特征提取
    data['cnt'] = 1

    # 主键
    keys = ['User_id', 'Coupon_id']
    # 特征名前缀,由history_field和主键组成
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    # 返回的特征数据集
    uc_feat = label_field[keys].drop_duplicates(keep='first')

    # 历史上用户领取该优惠券次数
    # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = pd.pivot_table(data, index=keys, values='cnt', aggfunc=len)
    # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    pivot = pd.DataFrame(pivot).rename(
        columns={'cnt': prefixs + 'receive_cnt'}).reset_index()
    # 将id列与特征列左连
    uc_feat = pd.merge(uc_feat, pivot, on=keys, how='outer')
    # 缺失值填充为0,最好加上参数downcast='infer',不然可能会改变DataFrame某些列中元素的类型
    uc_feat.fillna(0, downcast='infer', inplace=True)

    # 历史上用户消费该优惠券次数
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(
        columns={'cnt': prefixs + 'receive_and_consume_cnt'}).reset_index()
    uc_feat = pd.merge(uc_feat, pivot, on=keys, how='left')
    uc_feat.fillna(0, downcast='infer', inplace=True)

    # 历史上用户未消费该优惠券次数
    uc_feat['cnt'] = uc_feat.apply(lambda x:
                                    x['history_field_User_id_Coupon_id_receive_cnt']
                                    -x['history_field_User_id_Coupon_id_receive_and_consume_cnt'], axis=1)
    uc_feat.rename(columns={'cnt': prefixs + 'receive_and_unconsume_rate'}, inplace=True)

    # 历史上用户对该优惠券的核销率
    uc_feat['rate']=uc_feat.apply(lambda x:
                                x['history_field_User_id_Coupon_id_receive_and_consume_cnt']
                                /x['history_field_User_id_Coupon_id_receive_cnt']
                                if x['history_field_User_id_Coupon_id_receive_cnt'] > 0
                                else 0, axis=1)
    uc_feat.rename(columns={'rate': 'history_field_User_id_Coupon_id_receive_and_consume_rate'},inplace=True)

    #历史上该优惠卷被领取次数
    keys=['Coupon_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data, index=['Coupon_id'], values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(
        columns={'cnt': prefixs+'receive_cnt'}).reset_index()
    uc_feat = pd.merge(uc_feat, pivot, on=keys, how='left')
    uc_feat.fillna(0, downcast='infer', inplace=True)

    #历史上该优惠卷被消费次数
    keys = ['Coupon_id']
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    pivot = pd.pivot_table(data[data['Date'].map(lambda x: str(x) != 'null')], index=keys, values='cnt',
                           aggfunc=len)
    pivot = pd.DataFrame(pivot).rename(
        columns={'cnt': prefixs+'receive_and_consume_cnt'}).reset_index()
    uc_feat = pd.merge(uc_feat, pivot, on=keys, how='left')
    uc_feat.fillna(0, downcast='infer', inplace=True)

    #历史上该优惠卷未被消费次数
    uc_feat['cnt'] = uc_feat.apply(lambda x:
                                   x['history_field_Coupon_id_receive_cnt']
                                   - x['history_field_Coupon_id_receive_and_consume_cnt'], axis=1)
    uc_feat.rename(columns={'cnt': 'history_field_Coupon_id_receive_and_unconsume_cnt'}, inplace=True)

    #历史上该优惠卷被消费率
    uc_feat['rate'] = uc_feat.apply(lambda x:
                                    x['history_field_Coupon_id_receive_and_consume_cnt']
                                    / x['history_field_Coupon_id_receive_cnt']
                                    if x['history_field_Coupon_id_receive_cnt'] > 0
                                    else 0, axis=1)
    uc_feat.rename(columns={'rate': 'history_field_Coupon_id_receive_and_consume_rate'}, inplace=True)


    # 返回
    return uc_feat

def get_history_field_feature(label_field, history_field):
    # 用户特征
    uc_feat = get_history_field_user_coupon_feature(label_field, history_field)
    # 添加特征
    history_feat = label_field.copy()
    # 添加用户特征
    history_feat = pd.merge(history_feat, uc_feat, on=['User_id', 'Coupon_id'], how='left')
    # 返回
    return history_feat

def get_dataset(history_field, middle_field, label_field):
    """构造数据集

    Args:

    Returns:

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
        [history_feat, middle_feat.drop(share_characters, axis=1)], axis=1,join_axes=[history_feat.index])
    dataset = pd.concat(
        [dataset, label_feat.drop(share_characters, axis=1)], axis=1,join_axes=[dataset.index])

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


def model_xgb(train, test):
    """xgb模型

    Args:

    Returns:

    """
    # xgb参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'nthread': 4,
              'silent': 0,
              'eta': 0.01,
              'max_depth': 5,
              'min_child_weight': 1,
              'gamma': 0,
              'lambda': 1,
              'colsample_bylevel': 0.7,
              'colsample_bytree': 0.7,
              'subsample': 0.9,
              'scale_pos_weight': 1}
    # 数据集
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1), label=train['label'])
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1))
    # 训练
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=563, evals=watchlist)
    # 预测
    predict = model.predict(dtest)
    # 处理结果
    predict = pd.DataFrame(predict, columns=['prob'])
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], predict], axis=1)
    # 特征重要性
    feat_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feat_importance['feature_name'] = model.get_score().keys()
    feat_importance['importance'] = model.get_score().values()
    feat_importance.sort_values(['importance'], ascending=False, inplace=True)
    # 返回
    return result, feat_importance


if __name__ == '__main__':
    # # 源数据
    # off_train = pd.read_csv(r'../test8/input_files/ccf_offline_stage1_train.csv')
    # off_test = pd.read_csv(r'../test8/input_files/ccf_offline_stage1_test_revised.csv')
    # # 预处理
    # off_train = prepare(off_train)
    # off_test = prepare(off_test)
    # # 打标
    # off_train = get_label(off_train)
    #
    # # 划分区间
    # # 训练集历史区间、中间区间、标签区间
    # train_history_field = off_train[
    #     off_train['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501)
    # train_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/5/1', periods=15))]  # [20160501,20160516)
    # train_label_field = off_train[
    #     off_train['date_received'].isin(pd.date_range('2016/5/16', periods=31))]  # [20160516,20160616)
    # # 验证集历史区间、中间区间、标签区间
    # validate_history_field = off_train[
    #     off_train['date_received'].isin(pd.date_range('2016/1/16', periods=60))]  # [20160116,20160316)
    # validate_middle_field = off_train[
    #     off_train['date'].isin(pd.date_range('2016/3/16', periods=15))]  # [20160316,20160331)
    # validate_label_field = off_train[
    #     off_train['date_received'].isin(pd.date_range('2016/3/31', periods=31))]  # [20160331,20160501)
    # # 测试集历史区间、中间区间、标签区间
    # test_history_field = off_train[
    #     off_train['date_received'].isin(pd.date_range('2016/4/17', periods=60))]  # [20160417,20160616)
    # test_middle_field = off_train[off_train['date'].isin(pd.date_range('2016/6/16', periods=15))]  # [20160616,20160701)
    # test_label_field = off_test.copy()  # [20160701,20160801)
    #
    # # 构造训练集、验证集、测试集
    # print('构造训练集')
    # train = get_dataset(train_history_field, train_middle_field, train_label_field)
    # print('构造验证集')
    # validate = get_dataset(validate_history_field, validate_middle_field, validate_label_field)
    # print('构造测试集')
    # test = get_dataset(test_history_field, test_middle_field, test_label_field)
    #
    # # 保存数据集
    # train.to_csv("./prepared_dataset/train.csv")
    # validate.to_csv("./prepared_dataset/validate.csv")
    # test.to_csv("./prepared_dataset/test.csv")

    train = pd.read_csv(r'./prepared_dataset/train.csv')
    validate = pd.read_csv("./prepared_dataset/validate.csv")
    test = pd.read_csv("./prepared_dataset/test.csv")

    # 线上训练
    big_train = pd.concat([train, validate], axis=0)
    result, feat_importance = model_xgb(big_train, test)
    # 保存
    result.to_csv(r'./output_files/test.csv', index=False, header=None)

