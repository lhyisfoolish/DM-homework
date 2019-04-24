# coding: utf-8
# 1. 问题描述
# 本次作业中，将选择2个数据集进行探索性分析与预处理。
# 2. 数据说明
# 数据集为Wine Reviews中的winemag-data-130k-v2
# 3. 数据分析要求
# 3.1 数据可视化和摘要
# 数据摘要
# 对标称属性，给出每个可能取值的频数，
# 数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数。
# 数据的可视化
# 针对数值属性，

# 绘制直方图，用qq图检验其分布是否为正态分布。
# 绘制盒图，对离群值进行识别
# 3.2 数据缺失的处理
# 观察数据集中缺失数据，分析其缺失的原因。

# 分别使用下列四种策略对缺失值进行处理:

# 将缺失部分剔除
# 用最高频率值来填补缺失值
# 通过属性的相关关系来填补缺失值
# 通过数据对象之间的相似性来填补缺失值
# 处理后，可视化地对比新旧数据集。

# ----------------------------------------------------------

### BEGIN HERE

## Dataset
# Wine Reviews
# https://www.kaggle.com/zynicide/wine-reviews

## Columns
# country: where the wine was produced
# description: comments of the taster
# designation: name of the wine
# points: how much points it got by the taster(0-100)
# price: price of the wine(numbers)
# province: more detailed information of its origin
# region_1: more detailed information of its origin
# region_2: more detailed information of its origin
# taster_name: taster's name
# taster_twitter_handle: taster's twitter
# title: more details about the wine, including name, year, origin
# variety: wine's kind
# winery: which winery made the wine

## import basic packages
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

FILE_PATH = './winemag-data-130k-v2.csv'

## create DataFrame from csv file
df = pd.read_csv(FILE_PATH, index_col=0)
len_total = df.shape[0]
print('It has {0} items in total.'.format(len_total))

## Abstract
# country:
print(df['country'].value_counts())
# points:
print('points:')
print('max: {0}'.format(df['points'].max()))
print('min: {0}'.format(df['points'].min()))
print('mean: {0}'.format(df['points'].mean()))
print('median: {0}'.format(df['points'].median()))
print('quantile 1: {0}'.format(df['points'].quantile(0.25)))
print('quantile 2: {0}'.format(df['points'].quantile(0.5)))
print('quantile 3: {0}'.format(df['points'].quantile(0.75)))
print('missing: {0}'.format(df.shape[0] - df['points'].count()))
# price:
print('price:')
print('max: {0}'.format(df['price'].max()))
print('min: {0}'.format(df['price'].min()))
print('mean: {0}'.format(df['price'].mean()))
print('median: {0}'.format(df['price'].median()))
print('quantile 1: {0}'.format(df['price'].quantile(0.25)))
print('quantile 2: {0}'.format(df['price'].quantile(0.5)))
print('quantile 3: {0}'.format(df['price'].quantile(0.75)))
print('missing: {0}'.format(df.shape[0] - df['price'].count()))
# taster name:
print(df['taster_name'].value_counts())
# variety:
print(df['variety'].value_counts())


# define a method to process related figures
def show_figures(df):
    # Histogram of points
    histogram_points = df['points'].hist(
        grid=False, xlabelsize=0.4, bins=20).get_figure()
    plt.title('histogram of points')
    plt.show()
    # Q-Q plot of points
    stats.probplot(df['points'], dist='norm', plot=plt)
    plt.title('Q-Q plot of points')
    plt.show()

    # Histogram of price
    # ############ CAUTION ############
    # since price values mostly falls in (0,100), we only show the histogram of prices under 100.
    histogram_price = df['price'][df['price'] < 100].hist(
        grid=False, xlabelsize=10, bins=20).get_figure()
    plt.title('histogram of price')
    plt.show()
    # Q-Q plot of price
    stats.probplot(df['price'][df['price'] < 100], dist='norm', plot=plt)
    plt.title('Q-Q plot of price')
    plt.show()

    # Boxplot of points
    boxplot_points = df.boxplot(column=['points'])
    plt.title('Boxplot of points')
    plt.show()
    # Boxplot of price
    boxplot_price = df.boxplot(column=['price'])
    plt.title('Boxplot of price')
    plt.show()


show_figures(df)

## Begin processing
# 1.将缺失部分剔除
# delete items with empty price value(Attribute points has no empty value)
df1 = df.dropna(subset=['price'])
print('Now we have {0} items in total.'.format(df1.shape[0]))
print('We deleted {0} items.'.format(len_total - df1.shape[0]))
show_figures(df1)

# 2.用最高频率值来填补缺失值
# fillna price (Attribute points has no empty value)
df2 = df.copy()
df2_price = df['price'].fillna(df['price'].mode()[0])
df2['price'] = df2_price
show_figures(df2)

# 3.通过属性的相关关系来填补缺失值
# 由于缺失属性price和其他属性没有潜在联系，此部分略过，不做处理。

# 4.通过数据对象之间的相似性来填补缺失值
# price <-> designation:
# if designation_a == designation_b then fill price_b with price_a

df4 = df.copy()

# # firstly, we delete the rows with empty designation.
print('before delete: {0}'.format(df4.shape[0]))
df4 = df4.loc[df4['designation'].isnull() == False]
print('after delete: {0}'.format(df4.shape[0]))

# ############ CAUTION ############
# it tooks minutes to fill
while len(df4[df4['price'].isnull()]) > 0:

    df_missing = df4[df4['price'].isnull()]

    designation = df_missing.iloc[0]['designation']
    country = df_missing.iloc[0]['country']
    region_1 = df_missing.iloc[0]['region_1']
    # filter: designation, country, and region_1
    price = df[(df['designation'] == designation)
               & (df['country'] == country)
               & (df['region_1'] == region_1)
               & (df['price'].isnull() == False)]['price']
    if len(price) == 0:
        # if we can't find valid items then delete it/them
        df4 = df4.loc[df['designation'] != designation]
    else:
        # 取众数来填充空值
        x = price.mode()[0]
        tmp = df4.loc[(df4['designation'] == designation)
                      & (df4['price'].isnull())]  # type: object
        tmp['price'] = x
        df4.update(tmp)

show_figures(df4)