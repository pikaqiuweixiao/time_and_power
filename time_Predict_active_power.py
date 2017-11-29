# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV #网格搜索模型
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.ensemble import GradientBoostingRegressor

import warnings
#防止中文乱码
mpl.rcParams['font.sans-serif']=['simhei']
mpl.rcParams['axes.unicode_minus']=False
#过滤某警告
warnings.filterwarnings(action='ignore',category=ConvergenceWarning)


path='datas\household_power_consumption_1000.txt'
names=['Date','Time','Global_active_power',
       'Global_reactive_power','Voltage','Global_intensity',
       'Sub_metering_1','Sub_metering_2','Sub_metering_3']
#读取数据
df=pd.read_csv(path,sep=';')#sep分隔符为；
#print(df)
#print(df.head())#看前几行，默认前5行

#空值的处理
new_df=df.replace('?',np.nan)
datas=new_df.dropna(how='any')#有nan值就删除一行
#print(datas.describe())#观察数据的多种统计指标

#创建一个时间字符串格式化,把字符串连接，转换为时间元祖
def data_format(dt):
    import time
    t=time.strptime(' '.join(dt),'%d/%m/%Y %H:%M:%S')
    if t.tm_wday == 5 and t.tm_wday == 6:
        worktime = 0
    else:
        worktime = 1

    if (t.tm_hour >= 0 and t.tm_hour <= 6):
        peak_time = 1
    elif (t.tm_hour >= 6 and t.tm_hour <=17) or (t.tm_hour>=23 and  t.tm_hour<=24):
        peak_time = 2
    else:
        peak_time = 3

    if (t.tm_mon >= 5 and t.tm_mon <= 10):
        peak_mon=1
    else:
        peak_mon=2


    return (t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_wday, peak_mon,worktime, peak_time)

#获取X,Y变量，把时间转换为数值型的连续变量


X=datas[names[0:2]] #线性回归，参数多，分开比较好
X=X.apply(lambda x:pd.Series(data_format(x)),axis=1)

print(X.head())
Y=datas[names[2]]
print(Y.head(5))

models=[
    Pipeline([
        ('ss',StandardScaler()),
        ('poly',PolynomialFeatures()),
        ('linear',RidgeCV(alphas=np.logspace(-3,1,20)))
    ]),
    Pipeline([
            ('ss',StandardScaler()),
            ('poly',PolynomialFeatures()),
            ('linear',LassoCV(alphas=np.logspace(-3,1,20)))
        ])
]

parameters={
    'poly__degree':[1,2,3],
    'poly__interaction_only':[True,False], #是否形成交互项x1*x2
    'poly__include_bias':[True,False],#多项式次数为0的特征作为线性模型的截距
    'linear__fit_intercept':[True,False], #线性模型截距
}
#划分训练集与测试集
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

ln_x_test=range(len(X_test))
#训练并画图比较
plt.figure(figsize=(16,8),facecolor='w')

#画出真实值
plt.plot(ln_x_test,Y_test,'r',label='真实值')

titles=['RidgeCV','LassoCV']
colors=['g','b']

for t in range(2):
    print('t=',t)
    model=GridSearchCV(models[t],param_grid=parameters,n_jobs=1,cv=5)
    model.fit(X_train,Y_train)

    print('%s算法，最优参数：'%titles[t],model.best_params_)
    print('%s算法，R^2=%.3f：'%(titles[t],model.best_score_))#交叉验证，把多次验证的平均值作为对算法精度的估计

    y_predict=model.predict(X_test)
    plt.plot(ln_x_test,y_predict,color=colors[t],label='%s算法估计值，$R^2$=%.3f'%(titles[t],model.best_score_))

plt.legend(loc='upper left')
plt.suptitle('时间与功率预测')
plt.grid(True)
#plt.show()


gbr=GradientBoostingRegressor()
gbr.fit(X_train,Y_train)
y_predict=gbr.predict(X_test)
print('梯度提升算法训练集准确率',gbr.score(X_train,Y_train))
print('梯度提升算法测试集准确率',gbr.score(X_test,Y_test))