import random
import numpy as np
from scipy.spatial.distance import cityblock
# Step1
class KMeans:
    def __init__(self,n_clusters=2,max_iter=100):
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.centroids = None
# Step2
    def fit_predict(self,X):
        # print(X)
        random_index =(random.sample(range(0,X.shape[0]),self.n_clusters)) #Step 2 Initialize the centroids
        self.centroids=X[random_index]
        # print(self.centroids)

        for i in range(self.max_iter):
            #assign clusters
            cluster_group=self.assign_clusters(X) #[0,1,1,1,0]
        # print(cluster_group.shape)
            old_centroids = self.centroids

            #move centroids 
            self.centroids=self.move_centroids(X,cluster_group)
        
            #check finish
            if (old_centroids == self.centroids).all():
                break    

        return cluster_group    

    def assign_clusters(self,X):
        cluster_group=[]
        distances = []

        for row in X:  #100 rows * 2 centroids ==200 distances
            for centroid in self.centroids:
                distances.append(cityblock(row,centroid))
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)
            distances.clear()        

        return np.array(cluster_group)        
    

    def move_centroids(self,X,cluster_group):
        new_centroids = []
        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group==type].mean(axis=0)) #new centroids

        return np.array(new_centroids)  