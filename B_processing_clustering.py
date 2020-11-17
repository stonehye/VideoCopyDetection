import torch
import glob
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
import random
import argparse


def load_features(path, limit):
    total_local_features = []

    features_list = glob.glob(os.path.join(path, '*'))
    random.shuffle(features_list)
    print("total: ", len(features_list))
    count=0
    for features in features_list:
        print(count+1); count+=1

        video = torch.load(features)
        for frames in video:
            total_local_features += frames
        if len(total_local_features) >= limit: break

    random.shuffle(total_local_features)
    return total_local_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='B. processing clustering')

    parser.add_argument('--feature_path', required=True,
                        default='/nfs_shared_/hkseok/features_local/multiple/vcdb_core-1fps-MobileNet_triplet_sum-5sec',
                        help="feature path")
    parser.add_argument('--n_clust', required=False, default=20000, help="# of cluster centroid")
    parser.add_argument('--model_path', required=True, default='model-mul-vcdb_core-1fps-MobileNet_triplet_sum-5sec-20000clusters.pkl')
    parser.add_argument('--n_features', required=False,
                        default=1225000, help="Limit number of local features required to build a clustering model")

    args = parser.parse_args()
    print(args)

    n_clust = int(args.n_clust)
    feature_DB = args.feature_path
    limit = int(args.n_features)
    total_local_features = load_features(feature_DB, limit)
    x = torch.cat(total_local_features).numpy()
    print(x.shape)

    kmeans = MiniBatchKMeans(n_clusters=n_clust, random_state=0, verbose=True).fit(x)
    pickle.dump(kmeans, open(args.model_path, "wb"))