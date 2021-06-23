import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import time

data = pd.read_excel('./data/413231030.xlsx')

data2 = data[['longitude', 'latitude']].values.reshape(-1, 2)
scaler = MinMaxScaler()
scaler_data = scaler.fit_transform(data2)
cols = ['longitude', 'latitude']
df = pd.DataFrame(scaler_data, columns=cols)
clustering = DBSCAN(eps=0.05, min_samples=500, algorithm='ball_tree', n_jobs=-1).fit(scaler_data)
label = clustering.labels_
df['label'] = label
df2 = df.groupby(df['label'])
for key, sub_df in df2:
    print(key)
    print(sub_df)

# df.plot.scatter(x='longitude', y='latitude', c='label', colormap='jet')
# plt.show()
