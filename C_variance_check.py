import glob
from sklearn.preprocessing import MinMaxScaler
import pickle
from pipeline2 import *
import argparse


parser = argparse.ArgumentParser(description='C. BOW encoding')

parser.add_argument('--n_clust', required=False, default=15000, help="# of cluster centroid")
parser.add_argument('--model_path', required=False, default='/nfs_shared_/hkseok/cluster/K/resnet50-localfeature-150002.pkl')
parser.add_argument('--feature_path', required=False,
                        default='/nfs_shared_/hkseok/features_local/multiple/vcdb_core-1fps-resnet50_sum-5sec2',
                        help="source feature path")

args = parser.parse_args()
print(args)

N_cluster = int(args.n_clust)
kmeans = pickle.load(open(args.model_path, "rb"))
kmeans.verbose = False
feature_list = glob.glob(os.path.join(args.feature_path, '*'))

bins = [0] * N_cluster

for idx, feature in enumerate(feature_list):
    videoname = os.path.basename(feature)
    print(feature, idx)
    local_features = torch.load(feature)

    video_vector = []
    for frame in local_features:
        x = torch.cat(frame).numpy()
        predict = kmeans.predict(x)

        unique, counts = np.unique(predict, return_counts=True)
        for uniq, cnts in zip(unique, counts):
            bins[uniq] += cnts
print(bins)

name = '25000_again'

with open(f'{name}.pkl', 'wb') as fw:
    pickle.dump(bins, fw)

import csv
with open(f'{name}.csv', 'w') as fw:
    wr = csv.writer(fw)
    wr.writerow(['bin', 'count'])
    for idx, bin in enumerate(bins):
        wr.writerow([idx, bin])

