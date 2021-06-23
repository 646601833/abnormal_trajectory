import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import os

path = '../data/'
files = os.listdir(path)
for file in files:
    file_path = path + file
    data = pd.read_excel(file_path)
    df = data[['longitude', 'latitude']]
    # df.plot(x='longitude', y='latitude')
    scaler = MinMaxScaler()
    scale_data = scaler.fit_transform(df.values.reshape(-1, 2))
    cluster = DBSCAN(eps=0.05, min_samples=100, algorithm='ball_tree', n_jobs=-1).fit(scale_data)
    df.loc[:, 'label'] = cluster.labels_
    df.plot.scatter(x='longitude', y='latitude', c='label', colormap='viridis')
plt.show()