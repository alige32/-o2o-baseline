import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

# 读取数据
dfoff = pd.read_csv('./ccf_offline_stage1_train.csv')
dfon = pd.read_csv('./ccf_online_stage1_train.csv')
dftest = pd.read_csv('./ccf_offline_stage1_test_revised.csv')

"""
特征工程
"""


# 处理折扣
def pre_discount(x):
    if pd.isnull(x):
        return 1
    elif ':' in str(x):
        discount_list = x.split(':')
        return (int(discount_list[0]) - int(discount_list[1])) / int(discount_list[0])
    else:
        return float(x)


# 满
def getDiscountFull(x):
    if ':' in str(x):
        lis = x.split(':')
        return int(lis[0])
    else:
        return 0


# 减
def getDiscountReduction(x):
    if ':' in str(x):
        lis = x.split(':')
        return int(lis[1])
    else:
        return 0


# weekday
def weekday(x):
    wd = pd.to_datetime(str(int(x)), format="%Y/%m/%d").weekday()
    if wd == 0:
        return 7
    else:
        return wd


# 处理标签
def handle_label(x):
    if not pd.isnull(x['Date']):
        limit = pd.to_datetime(str(int(x['Date'])), format="%Y/%m/%d") - pd.to_datetime(str(int(x['Date_received'])),
                                                                                        format="%Y/%m/%d")
        if pd.Timedelta(15, 'D') >= limit:
            return 1
    return 0


def handle_train_datas(datas):
    # 距离转换为int类型
    datas['Distance'] = datas.Distance.map(lambda x: -1 if pd.isnull(x) else int(x))
    # 折扣转换为折扣率，并把满和减单独作为一列
    datas['Discount_rate'] = datas.Discount_rate.fillna(1)
    datas['rate'] = datas.Discount_rate.map(pre_discount)
    datas['full'] = datas.Discount_rate.map(getDiscountFull)
    datas['reduction'] = datas.Discount_rate.map(getDiscountReduction)
    # 领取日期，提取星期和是否周末单独作一列
    datas.dropna(subset=['Date_received'], inplace=True)
    datas['weekday'] = datas.Date_received.map(weekday)
    datas['weekend'] = datas.weekday.map(lambda x: 1 if x == 7 or x == 6 else 0)
    label = datas.apply(handle_label, axis=1)
    pre_datas = datas.loc[:, ['rate', 'full', 'reduction', 'weekday', 'weekend']]
    return pre_datas, label


def handle_test_datas(datas):
    # 距离转换为int类型
    datas['Distance'] = datas.Distance.map(lambda x: -1 if pd.isnull(x) else int(x))
    # 折扣转换为折扣率，并把满和减单独作为一列
    datas['Discount_rate'] = datas.Discount_rate.fillna(1)
    datas['rate'] = datas.Discount_rate.map(pre_discount)
    datas['full'] = datas.Discount_rate.map(getDiscountFull)
    datas['reduction'] = datas.Discount_rate.map(getDiscountReduction)
    # 领取日期，提取星期和是否周末单独作一列
    datas.dropna(subset=['Date_received'], inplace=True)
    datas['weekday'] = datas.Date_received.map(weekday)
    datas['weekend'] = datas.weekday.map(lambda x: 1 if x == 7 or x == 6 else 0)
    #     label=datas.apply(handle_label,axis=1)
    pre_datas = datas.loc[:, ['rate', 'full', 'reduction', 'weekday', 'weekend']]
    return pre_datas


def model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=18)
    XGB = XGBClassifier()
    xgb_param = {
        'n_estimators': 30,
        'max_depth': range(2, 5, 1)
    }
    gc = GridSearchCV(XGB, param_grid=xgb_param, cv=5)
    gc.fit(x_train, y_train)
    return gc


if __name__ == '__main__':
    pre_datas, Y = handle_train_datas(dfoff)
    pre_test_datas = handle_test_datas(dftest)

    model = model(pre_datas, Y)

    # 整理提交数据
    pred = model.predict_proba(pre_test_datas)
    sub_test = dftest[['User_id', 'Coupon_id', 'Date_received']].copy()
    sub_test['label'] = pred[:, 1]
    sub_test.to_csv('submit.csv', index=False, header=False)
