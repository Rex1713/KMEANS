from sklearn.datasets import make_blobs
from kmeans import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

centroids=[(-5,-5),(5,5),(-2.5,2.5)]
cluster_std=[1,1,1]


X,y = make_blobs(n_samples=100,cluster_std=cluster_std,centers=centroids,n_features=2,random_state=2)


km=KMeans(n_clusters=3,max_iter=100)
y_means = km.fit_predict(X)

from sklearn.metrics import silhouette_score
print(silhouette_score(X,y_means))

plt.scatter(X[y_means== 0,0],X[y_means==0,1],color='red')
plt.scatter(X[y_means== 1,0],X[y_means==1,1],color='blue')

plt.scatter(X[y_means== 2,0],X[y_means==2,1],color='green')
plt.show()


# plt.scatter(X[:,0],X[:,1])
# plt.show()
