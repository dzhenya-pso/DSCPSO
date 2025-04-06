# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 22:57:46 2024
@author:Slogan
About:K-means Phenotype Entropy fitness landscape PSO
"""
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.datasets import make_blobs
from scipy.stats import qmc


# from CEC2013benchmarkfunctions.opfunu.CEC13 import CEC13  # 导入测试函数集

# np.random.seed(42)


class DSCPSO():
    def __init__(self, fitness, fbias, fun_num=1, D=30, P=20, G=500, ub=1, lb=0,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, k=0.2):
        self.fitness = fitness  # 适应度
        self.D = D  # 搜索空间维数
        self.P = P  # 种群规模
        self.G = G  # 最大迭代次数
        self.ub = ub * np.ones([self.P, self.D])  # 上限
        self.lb = lb * np.ones([self.P, self.D])  # 下限
        self.fbias = fbias
        self.num = fun_num
        self.w_max = w_max  # 最大惯性权重
        self.w_min = w_min  # 最小惯性权重
        self.w = w_max  # 初始时惯性权重最大
        self.c1 = c1  # 加速因子1
        self.c2 = c2  # 加速因子2
        self.Cgmin = 0.5
        self.Cgmax = 2.5
        self.Cpmin = 0.5
        self.Cpmax = 2.5
        self.k = k
        self.v_max = self.k * (self.ub - self.lb)  # 初始化最大速度

        self.pbest_X = np.zeros([self.P, self.D])  # 初始化局部最佳位置
        self.pbest_F = np.zeros([self.P]) + np.inf  # 初始化局部最佳位置的适应度
        self.gbest_X = np.zeros([self.D])  # 初始化全局最佳位置
        self.gbest_F = np.inf  # 初始化全局最佳位置的适应度
        self.loss_curve = np.zeros(self.G)  # 最佳适应度

    def opt(self):
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])  # 随机初始化粒子群位置
        self.V = np.zeros([self.P, self.D])  # 随机初始化粒子群速度

        # # 迭代

        for self.g in range(self.G):

            # 计算每个粒子的适应度值
            self.F = self.fitness(self.X) - self.fbias
            # 更新最佳解
            mask = self.F < self.pbest_F  # 如果当前粒子的适应度好于局部最佳粒子
            self.pbest_X[mask] = self.X[mask].copy()  # 则将其作为局部最佳位置（历史最佳）
            self.pbest_F[mask] = self.F[mask].copy()

            if np.min(self.F) < self.gbest_F:  # 如果当前最佳粒子的适应度好于全局最佳粒子
                idx = self.F.argmin()
                self.gbest_X = self.X[idx].copy()  # 则将其作为全局最佳位置
                self.gbest_F = self.F.min()

            # 收敛曲线
            self.loss_curve[self.g] = self.gbest_F

            # 进化状态估计
            self.FLStrategy()

            # 更新
            r1 = np.random.uniform(size=[self.P, self.D])
            r2 = np.random.uniform(size=[self.P, self.D])
            # Update the w of PSO using ranking information
            # Rank = np.argsort(self.F)
            # self.w = self.w_min + (Rank / self.P) * (self.w_max - self.w_min)

            self.V = self.w * self.V + self.c1 * (self.pbest_X - self.X) * r1 \
                     + self.c2 * (self.gbest_X - self.X) * r2  # 公式（1）
            self.V = np.clip(self.V, -self.v_max, self.v_max)  # 边界处理

            self.X = self.X + self.V  # 公式（2）
            self.X = np.clip(self.X, self.lb, self.ub)  # 边界处理

    def plot_curve(self):
        plt.figure()
        plt.title('loss curve [' + str(round(self.loss_curve[-1], 3)) + ']')
        plt.plot(self.loss_curve, label=f'CEC2017fun_{self.num} loss')
        plt.grid()
        plt.legend()
        plt.show()

    def FLStrategy(self):
        # 初始化uS
        uS1, uS2, uS3, uS4 = 0, 0, 0, 0
        Current_State = 'S1'  # 默认值
        # 步骤1
        # 计算一个进化因子f
        f, indices_min, indices_max = self.find_optimal_k()  # 寻找最优簇个数、不同簇下的种群个数、计算信息熵
        if 0.0 <= f <= 0.22:
            Current_State = 'S1'
        elif 0.22 < f <= 0.55:
            Current_State = 'S2'
        elif 0.55 < f <= 0.78:
            Current_State = 'S3'
        elif 0.78 < f <= 1.0:
            Current_State = 'S4'

        Final_State = Current_State

        # 加速因子的控制
        delta = np.random.uniform(low=0.05, high=0.1, size=2)  # 在[0.05,0.1]内均匀生成的随机值δ

        if Final_State == 'S4':  # 策略1：在探索状态下增大c1，减小c2
            self.c1 = self.c1 + delta[0]
            self.c2 = self.c2 - delta[1]
            # self.chushihua(indices_min)
        elif Final_State == 'S2':  # 策略2：AWPSO
            self.c1 = self.F_c1()
            self.c2 = self.F_c2()
        elif Final_State == 'S1':  # 策略3：RPSO
            self.c1, self.c2 = self.sigma()
            self.ElitistLearningStrategy(indices_min)
        elif Final_State == 'S3':  # 策略4：在跳出状态下减小c1，增大c2
            self.c1 = self.c1 - delta[0]
            self.c2 = self.c2 + delta[1]


        self.c1 = np.clip(self.c1, 1.5, 2.5)  # 选择区间[1.5,2.5]来收紧c1和c2
        self.c2 = np.clip(self.c2, 1.5, 2.5)

        # 令ω跟随进化状态，使用Sinusoidal映射ω(f)
        self.w = 2.3 * np.sin(np.pi * f) * f * f  # Sinusoidal混沌映射
        self.w = np.clip(self.w, self.w_min, self.w_max)

    def ElitistLearningStrategy(self, indices_min):
        # P = self.gbest_X.copy()
        P = np.sum(self.X[indices_min], axis=0) / len(indices_min)  # 如果按行求和后取均值
        d = np.random.randint(low=0, high=self.D)

        mu = 0
        sigma = 1 - 0.9 * self.g / self.G  # 公式（22）
        # ELS 随机选择gBest的历史最佳位置的某一维
        P[d] = P[d] + (self.ub[0, d] - self.lb[0, d]) * np.random.normal(mu, sigma ** 2)  # 公式（22）

        # 判断是否超出界限
        P = np.clip(P, self.lb[0], self.ub[0])
        # 转化成一行dim维的矩阵
        matrix = np.array(P).reshape(1, -1)
        # v = CEC13(matrix, self.num) - self.fbias
        v = self.fitness(matrix) - self.fbias

        if v < self.gbest_F:
            self.gbest_X = P.copy()
            self.gbest_F = v.copy()
        elif v < self.F.max():
            idx = self.F.argmax()
            self.X[idx] = P.copy()
            self.F[idx] = v.copy()

    def F_c1(self):
        a = 0.0000035 * (self.ub - self.lb)
        b = 0.5
        c = 0
        d = 1.5
        result = b / (1 + np.exp(-a * ((self.pbest_X - self.X) - c))) + d
        return result

    def F_c2(self):
        a = 0.0000035 * (self.ub - self.lb)
        b = 0.5
        c = 0
        d = 1.5
        result = b / (1 + np.exp(-a * ((self.gbest_X - self.X) - c))) + d
        return result

    def sigma(self):
        pdf = norm.pdf(self.g, loc=0, scale=0.07)
        c1 = self.Cpmin + ((self.Cpmax - self.Cpmin) *
                           (self.G - self.g)) / self.G + pdf
        c2 = self.Cgmin + ((self.Cgmax - self.Cgmin) *
                           (self.G - self.g)) / self.G + pdf
        return c1, c2

    def find_optimal_k(self):
        # 使用肘部法则计算不同簇数下的SSE
        k_values = range(1, 11)
        sse = []
        optimal_k = None
        self.F = self.F.reshape(-1, 1)  # 转换为二维数组

        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0)  # 设置 n_init='auto'
            kmeans.fit(self.F)
            sse.append(kmeans.inertia_)

            if len(sse) > 1:
                ratio = sse[-1] / (sse[-2] + 1e-10)
                if ratio >= 0.5:
                    optimal_k = k - 1
                    break

        if optimal_k is None:
            optimal_k = 1
        # 启用警告过滤
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        # 根据最佳k值进行K-means聚类
        kmeans = KMeans(n_clusters=optimal_k, n_init=optimal_k, random_state=0)
        kmeans.fit(self.F)

        # 统计每个簇中的样本数量
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique, counts))
        # 输出最佳簇数量
        # print(f'Optimal number of clusters selected: {optimal_k}')
        # 输出每个簇的样本数量
        # for cluster_num, count in cluster_counts.items():
        #     print(f'Cluster {cluster_num}: {count} samples')
        # ###### 获取最优值对应簇的下标 ######
        idx_min = (np.abs(self.F - self.gbest_F)).argmin()
        cluster_num_min = labels[idx_min]
        idx_max = (np.abs(self.F - self.gbest_F)).argmax()
        cluster_num_max = labels[idx_max]
        indices_min = np.where(labels == cluster_num_min)[0]
        indices_max = np.where(labels == cluster_num_max)[0]

        # 将 self.F 还原为原始形状（假设原始是一维数组）
        self.F = self.F.flatten()  # 转换为一维数组

        # 计算信息熵
        total_samples = sum(cluster_counts.values())
        probabilities = np.array(list(cluster_counts.values())) / total_samples
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # 加上小常数以避免 log(0)
        # return entropy
        # 信息熵归一化
        lambda_value = (1 / (1 + np.exp(-2.5 * (entropy - 1.5))))

        return lambda_value, indices_min, indices_max

