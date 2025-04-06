# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:52:46 2024
@author:Slogan
About:CEC2017测试函数
"""
"""
使用方法：
首先导入测试函数 -> from cec2017_py_master.cec2017 import functions as functions
functions包含了30个函数的头名字f1~f30
functions.f1(x)~functions.f30(x)或f=functions.f1~functions.f30-->f(x)
或for f in functions.all_functions:
    fitness=f(x),或者传递fitness-->fitness(x)来计算适应度值
f(x)中的x为一个numpy矩阵，行代表一个个体，也就是一个解决方案，列代表个体维度。
i为一个int型的数字，代表第几个测试函数。
### 下面为各个模块的使用方法
1、simple包含f1~f10:
首先导入测试函数:
from cec2017_py_master.cec2017 import simple as simple
simple.f1(x)~simple.f10(x)或f=simple.f1~simple.f10-->f(x)
或for f in simple.all_functions:
    fitness=f(x),或者传递fitness-->fitness(x)来计算适应度值
2、hybrid包含f11~f20:
首先导入测试函数:
from cec2017_py_master.cec2017 import hybrid as hybrid
hybrid.f11(x)~hybrid.f20(x)
或f=hybrid.f11~hybrid.f20-->f(x)
或for f in hybrid.all_functions:
    fitness=f(x),或者传递fitness-->fitness(x)来计算适应度值
3、composition包含f21~f30:
首先导入测试函数:
from cec2017_py_master.cec2017 import composition as composition
composition.f21(x)~composition.f30(x)
或f=composition.f21~composition.f30-->f(x)
或for f in hybrid.all_functions:
    fitness=f(x),或者传递fitness-->fitness(x)来计算适应度值
"""
import math, time
import pandas as pd
import numpy as np
# evaluate a specific function a few times with one sample
from cec2017_py_master.cec2017 import functions as functions
from cec2017_py_master.optimizers import DSCPSO

Algorithm_name = "DSCPSO"
# CEC2017最优值偏移量
fbias=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
       1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
       2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800,
       2900, 3000]

column_list = ['CEC2017Fun_1', 'CEC2017Fun_2', 'CEC2017Fun_3', 'CEC2017Fun_4',
                 'CEC2017Fun_5', 'CEC2017Fun_6', 'CEC2017Fun_7', 'CEC2017Fun_8',
                 'CEC2017Fun_9', 'CEC2017Fun_10', 'CEC2017Fun_11', 'CEC2017Fun_12',
                 'CEC2017Fun_13', 'CEC2017Fun_14', 'CEC2017Fun_15', 'CEC2017Fun_16',
                 'CEC2017Fun_17', 'CEC2017Fun_18', 'CEC2017Fun_19', 'CEC2017Fun_20',
                 'CEC2017Fun_21', 'CEC2017Fun_22', 'CEC2017Fun_23', 'CEC2017Fun_24',
                 'CEC2017Fun_25', 'CEC2017Fun_26', 'CEC2017Fun_27', 'CEC2017Fun_28',
                 'CEC2017Fun_29', 'CEC2017Fun_30']
# 设置参数
dim, lb, ub = 30, -100, 100    # dim->维度, lb->下限, ub->上限
PopSize = 100    # 2 ** (4 + math.floor(math.log2(math.sqrt(dim))))  # 2*dim 或者 2^(4+floor(log2(sqrt(D)))) #种群数量
fes_max = 10000*dim  # 最大评价次数  pop_size*iter_max
Maxiter = math.ceil(fes_max / PopSize)  # 最大迭代次数
run_times = 30  # 运行次数
# individual = np.random.uniform(lb, ub, (PopSize, dim))  # 生成5个个体维度为10维的种群
table = pd.DataFrame(np.zeros([5, 30]), index=['avg', 'std', 'worst', 'best', 'time'])
# avg：均值；std：标准差；worst：最差值；best：最佳值；ideal：期望值；time：运行时间
loss_curves = np.zeros([Maxiter, 30])  # 最佳适应度
F_table = np.zeros([run_times, 30])

for f, i in zip(functions.all_functions, range(0, 30)):  # f1 to f30
    print(f"------ No.{i + 1} begin!------")
    for n in range(run_times):
        # fitness = CEC13(individual, i+1)  # 评价函数
        optimizer = DSCPSO.DSCPSO(fitness=f, fbias=fbias[i], fun_num=i+1,
                              D=dim, P=PopSize, G=Maxiter, ub=ub, lb=lb)
        st = time.time()
        optimizer.opt()
        ed = time.time()
        F_table[n, i] = optimizer.gbest_F
        table[i]['avg'] += optimizer.gbest_F
        table[i]['time'] += ed - st
        loss_curves[:, i] += optimizer.loss_curve
        print(f"第{n + 1}次运行结果为：{optimizer.gbest_F}")
        if n == run_times - 1:
            print(f"Function F{i + 1}:\n Avg. fitness = {table[i]['avg'] / run_times:.2e}({F_table[:, i].std():.2e})")
        # if n==run_times-1:
        #     optimizer.plot_curve()
        # print("种群适应度值为：", fitness)  # 得到了5个个体的适应度值

# 数据保存
loss_curves = loss_curves / run_times
loss_curves = pd.DataFrame(loss_curves)
loss_curves.columns = column_list
loss_curves.to_csv(f'loss_curves({Algorithm_name}_{dim}D).csv')

table.loc[['avg', 'time']] = table.loc[['avg', 'time']] / run_times
table.loc['worst'] = F_table.max(axis=0)
table.loc['best'] = F_table.min(axis=0)
table.loc['std'] = F_table.std(axis=0)
table.columns = column_list
table.to_csv(f'table({Algorithm_name}_{dim}D).csv')

# 保存所有结果
F_table = pd.DataFrame(F_table)
F_table.columns = column_list
F_table.to_csv(f'all/F_table({Algorithm_name}_{dim}D).csv')