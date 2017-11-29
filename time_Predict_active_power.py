# -*- coding:utf-8 -*-
'''
预测功率与电流的关系
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt

# 1.首先读取文件
path = 'datas\household_power_consumption.txt'
df = pd.read_csv(path, sep=';')

new_df=df.replace('?',np.nan)
datas=new_df.dropna(how='any')#有nan值就删除一行

# print(df.head())
names = ['Date', 'Time', 'Global_active_power',
         'Global_reactive_power', 'Voltage', 'Global_intensity',
         'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# 2.获取X,Y变量的值
# 首先x有两个特征x1,x2，有效功率与无效功率相加
X = datas[names[2:4]]
Y = datas[names[5]]
# print(X.head())
# print(Y.head())


# 3.进行训练集，测试集分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.000001, random_state=1)

# 4.进行标准化处理
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 5.建立线性回归模型,进行训练
lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_predict = lr.predict(X_test)
print('拟合优度是：{}'.format(lr.score(X_test, Y_test)))
print('参数列表：',lr.coef_)
# 预测值与真实值比较，画图
# 防止中文乱码

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(facecolor='w')
t = np.arange(len(X_test))
plt.plot(t, Y_test, color='r', label='真实值')
plt.plot(t, Y_predict, color='b', label='预测值')
plt.title('线性回归预测功率与电流的关系')
plt.legend(loc='best')
plt.grid(True)
plt.show()
