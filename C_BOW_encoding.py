import torch
import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pickle
import random
import sys

from pipeline2 import *


scaler = MinMaxScaler()

N_cluster = 20000
kmeans = pickle.load(open("model_400size_20000.pkl", "rb"))

# feature = '/nfs_shared_/hkseok/features_local/vcdb_core-1fps-mobilenet-5sec/00274a923e13506819bd273c694d10cfa07ce1ec.flv.pth'
# local_features = torch.load(feature)
# video_vector = []
# for frame in local_features:
#     x = torch.cat(frame).numpy()
#     predict = kmeans.predict(x)
#     frame_vector = np.array([np.count_nonzero(predict==i) for i in range(N_cluster)])
#     # TODO: frame_vector normalization
#     video_vector.append(torch.from_numpy(frame_vector))
# video_vector = torch.stack(video_vector, dim=0)
# print(video_vector.shape)


dst = '/nfs_shared_/hkseok/BOW/'
basename = 'vcdb_core-1fps-mobilenet-5sec-400size'
pth_dir = os.path.join(dst, basename) # pth file path
if not os.path.isdir(pth_dir):
    os.makedirs(pth_dir)

feature_list = glob.glob(os.path.join('/nfs_shared_/hkseok/features_local/single', basename, '*'))
for idx, feature in enumerate(feature_list):
    videoname = os.path.basename(feature)
    print(feature, idx)
    local_features = torch.load(feature)
    video_vector = []
    for frame in local_features:
        x = torch.cat(frame).numpy()
        predict = kmeans.predict(x)

        # BOW encoding
        frame_vector = np.array([np.count_nonzero(predict == i) for i in range(N_cluster)]).astype(np.float32)

        # minmax scalar
        scaler.fit(frame_vector.reshape(-1,1))
        transformed_vector = scaler.transform(frame_vector.reshape(-1,1))
        transformed_vector = transformed_vector.squeeze(-1)

        video_vector.append(torch.from_numpy(transformed_vector))

    video_vector = torch.stack(video_vector, dim=0)
    dst_path = os.path.join(pth_dir, videoname)
    torch.save(video_vector, dst_path)
    print(video_vector.shape)