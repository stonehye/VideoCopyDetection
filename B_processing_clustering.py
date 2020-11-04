import torch
import numpy as np
import glob
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import OneHotEncoder
import pickle
import random

n_clust = 20000

def load_features(path):
    total_local_features = []

    features_list = glob.glob(os.path.join(path, '*'))
    print(len(features_list))
    count=0
    for features in features_list:
        print(count+1)
        count+=1
        video = torch.load(features)

        for frames in video:
            total_local_features += frames

    random.shuffle(total_local_features)
    return total_local_features[:845000]


feature_DB = '/nfs_shared_/hkseok/features_local/single/vcdb_core_train-1fps-mobilenet-5sec-400size'
# feature_DB = 'testfeatures/'
total_local_features = load_features((feature_DB))
print(len(total_local_features))
x = torch.cat(total_local_features).numpy()
print(x.shape)

kmeans = MiniBatchKMeans(n_clusters=n_clust, random_state=0,verbose=True).fit(x)
pickle.dump(kmeans, open("model_400size_20000.pkl", "wb"))

# sample = np.random.randn(49, 1280).astype('f')
# predict = kmeans.predict(sample)
# print(predict)
#
# BOW = np.array([np.count_nonzero(predict==i) for i in range(n_clust)])
# print(BOW)
# print(predict)
# cluster_ids_x, cluster_centers = kmeans(
#     X=x, num_clusters=3, distance='euclidean', device=torch.device('cuda:0'))
#
# print(cluster_centers.size())
# print(cluster_centers)
