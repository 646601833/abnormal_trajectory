import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import math
import numpy as np
import KDTree


# 加载数据
def load_data(path):
    source_data = pd.read_excel(path)
    data = source_data[['longitude', 'latitude']].values.reshape(-1, 2)
    scaler = MinMaxScaler()
    data2 = scaler.fit_transform(data)
    return data2


# 计算两个向量之间的欧式距离
def dist(a, b):
    return math.sqrt(np.power(a - b, 2).sum())


def eps_neighbor(a, b, eps):
    """
    输入：向量A, 向量B
    输出：是否在eps范围内
    """
    return dist(a, b) < eps


def region_query(data, pointId, eps):
    """
    输入：数据集, 查询点id, 半径大小
    输出：在eps范围内的点的id
    """
    nPoints = data.shape[0]
    seeds = []
    for i in range(nPoints):
        if eps_neighbor(data[pointId, :], data[i, :], eps):
            seeds.append(i)
    return seeds


def num_in_area(data, pointId, eps):
    """
    输入：数据集, 查询点id, 半径大小
    输出：在eps范围内的点的个数
    """
    nPoints = data.shape[0]
    seeds = 0
    for i in range(nPoints):
        if eps_neighbor(data[pointId, :], data[i, :], eps):
            seeds += 1
    return seeds


def entropy(seeds):
    sum = np.array(seeds).sum()
    res = 0
    for i in range(len(seeds)):
        temp = (seeds(i)/sum) * np.log2(seeds(i)/sum)
        res += temp
    return -res


def num_neighbors(point, radius, data):
    neigh = NearestNeighbors(radius=radius, n_jobs=-1, algorithm='brute')
    neigh.fit(data)
    rng = neigh.radius_neighbors([point], return_distance=False)
    rng = np.asarray(rng[0])
    return rng


if __name__ == '__main__':
    path = '../data/413231030.xlsx'
    data = load_data(path)
    col = []
    for i in tqdm(range(len(data))):
        rng = num_neighbors(data[i], 0.05, data)
        col.append(len(rng))
    col = np.array(col)
    avg = col.sum()/len(col)
    print(avg)