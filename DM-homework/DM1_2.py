# -*- coding:UTF-8 -*-
# 1. 问题描述
# 本次作业中，将选择2个数据集进行探索性分析与预处理。
# 2. 数据说明
#数据集为Oakland Crime Statistics 2011 to 2016中2015年数据
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
# Oakland Crime Statistics 2011 to 2016
# https://www.kaggle.com/cityofoakland/oakland-crime-statistics-2011-to-2016

# Columns
# Agency
# Create Time
# Location
# Area Id
# Beat
# Priority
# Incident Type Id
# Incident Type Description
# Event Number
# Closed Time

# import basic packages
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

FILE_PATH = './records-for-2015.csv'

# create DataFrame from csv file
df = pd.read_csv(FILE_PATH, index_col=0)
len_total = df.shape[0]
print('It has {0} items in total.'.format(len_total))

## Abstract
# Location:
print(df['Location'].value_counts())
# Area Id:
print(df['Area Id'].value_counts())
# Beat:
print(df['Beat'].value_counts())
# Priority:
print(df['Priority'].value_counts())
# Incident Type Id:
print(df['Incident Type Id'].value_counts())
# Incident Type Description:
print(df['Incident Type Description'].value_counts())

# 由于此数据集中，没有数值属性的例子，所以无法绘制柱形图、盒图等。
# 将分析犯罪多发时间段、多发地点、多发类型；分析案件的平均处置时间。

# 1.多发时间段
# 分析月份；分析时间段（以一小时为一段）
# 提取Create Time, Closed Time中时间的月份并新建一列存储
df['Create Time formatted'] = pd.to_datetime(df['Create Time'])
df['Closed Time formatted'] = pd.to_datetime(df['Closed Time'])
df['Month'] = df['Create Time formatted'].dt.month
df['Hour'] = df['Create Time formatted'].dt.hour

# 绘制月份直方图

histogram_month = df['Month'].hist(
    grid=False, xlabelsize=1, bins=12).get_figure()
plt.title('histogram of month')
plt.show()

histogram_hour = df['Hour'].hist(
    grid=False, xlabelsize=1, bins=24).get_figure()
plt.title('histogram of hour')
plt.show()

# 2.多发地点(只统计TOP 10数据)
df['Location'].value_counts()[:10].plot(kind='barh', legend=True, title='Top 10 Locations')
plt.show()

# 3.多发类型(只统计TOP 10数据)
df['Incident Type Description'].value_counts()[:10].plot(kind='barh', legend=True, title='Top 10 Types')
plt.show()

# 4.平均处置时间
df['Handling Time'] = df['Closed Time formatted'] - df['Create Time formatted']
print(df['Handling Time'].mean())