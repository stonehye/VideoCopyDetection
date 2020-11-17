import glob
from sklearn.preprocessing import MinMaxScaler
import pickle
from pipeline2 import *
import argparse


parser = argparse.ArgumentParser(description='C. BOW encoding')

parser.add_argument('--n_clust', required=False, default=20000, help="# of cluster centroid")
parser.add_argument('--model_path', required=True, default='model-mul-vcdb_core-1fps-MobileNet_triplet_sum-5sec-20000clusters.pkl')
parser.add_argument('--feature_path', required=True,
                        default='/nfs_shared_/hkseok/features_local/multiple/vcdb_core-1fps-MobileNet_triplet_sum-5sec',
                        help="source feature path")
parser.add_argument('--BOW_path', required=True,
                        default='/nfs_shared_/hkseok/BOW/multiple/vcdb_core-1fps-MobileNet_triplet_sum-5sec',
                        help="dst feature path")

args = parser.parse_args()
print(args)

N_cluster = int(args.n_clust)
kmeans = pickle.load(open(args.model_path, "rb"))
feature_list = glob.glob(os.path.join(args.feature_path, '*'))
scaler = MinMaxScaler()
pth_dir = args.BOW_path
if not os.path.isdir(pth_dir):
    os.makedirs(pth_dir)

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
        frame_vector = scaler.transform(frame_vector.reshape(-1,1))
        frame_vector = frame_vector.squeeze(-1)

        video_vector.append(torch.from_numpy(frame_vector))

    video_vector = torch.stack(video_vector, dim=0)
    dst_path = os.path.join(pth_dir, videoname)
    torch.save(video_vector, dst_path)

    print(video_vector.shape)