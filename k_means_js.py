#!/ddn/home4/r2444/anaconda3/bin/python
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


data = pd.read_csv('mnist_train_js_patches_reduced.csv', header=None)
features= data.columns[1:]
label = data.columns[0]
data_features = data[features]
data_label = data[label]
data_features_np = data_features.to_numpy()
data_label_np = data_label.to_numpy()
data_features_np = data_features_np/ 255
#X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=300, random_state=0).fit(data_features_np)
print(len(data_features_np))
print(kmeans.cluster_centers_)

cluster_centers = pd.DataFrame(kmeans.cluster_centers_)
cluster_centers.to_csv('mnist_train_js_cluster_centers.csv', encoding='utf-8', index=False, header=None)
