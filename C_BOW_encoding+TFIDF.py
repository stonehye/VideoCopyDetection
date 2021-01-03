import argparse
import glob
import pickle
from math import log

from pipeline2 import *

parser = argparse.ArgumentParser(description='C. BOW encoding')

parser.add_argument('--n_clust', required=True, default=20000, help="# of cluster centroid")
parser.add_argument('--model_path', required=True, default='/nfs_shared_/hkseok/cluster/resnet50_conv3_4.pkl')
parser.add_argument('--feature_path', required=True,
                        default='/nfs_shared_/hkseok/features_local/multiple/resnet50_conv3_4',
                        help="source feature path")
parser.add_argument('--BOW_path', required=True,
                        default='/nfs_shared_/hkseok/BOW_tf-idf/multiple/resnet50_conv3_4',
                        help="dst feature path")

args = parser.parse_args()
print(args)

N_cluster = int(args.n_clust)
kmeans = pickle.load(open(args.model_path, "rb"))
kmeans.verbose = False
feature_list = glob.glob(os.path.join(args.feature_path, '*'))

pth_dir = args.BOW_path
if not os.path.isdir(pth_dir):
    os.makedirs(pth_dir)

docs = []
for idx, feature in enumerate(feature_list):
    print(feature, idx)
    videoname = os.path.basename(feature)
    local_features = torch.load(feature)

    for frame in local_features:
        x = torch.cat(frame).numpy()
        predict = kmeans.predict(x)
        docs.append(predict.tolist())

vocab = range(0, N_cluster)
N = len(docs)
print(N)


def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N / (df + 1))


def tfidf(t, d, idf):
    return d.count(t) * idf


IDF = []
for t in vocab:
    IDF.append(idf(t))

torch.save(IDF,'IDF.pth')
print(IDF)
#
# TFIDF = []
# for document in docs:
#     temp = [tfidf(j, document, IDF[j]) if j in document else 0. for j in range(N_cluster)]
#     TFIDF.append(torch.Tensor(temp))
# print(TFIDF)
#
# torch.save(TFIDF, "TFIDF.pth")

for idx, feature in enumerate(feature_list):
    print(feature, idx)
    videoname = os.path.basename(feature)
    dst_path = os.path.join(pth_dir, videoname)
    local_features = torch.load(feature)

    video_vector=[]
    for frame in local_features:
        x = torch.cat(frame).numpy()
        predict = kmeans.predict(x).tolist()
        temp = [tfidf(j, predict, IDF[j]) if j in predict else 0. for j in range(N_cluster)]
        video_vector.append(torch.Tensor(temp))

    video_vector = torch.stack(video_vector, dim=0)
    torch.save(video_vector, dst_path)
    print(video_vector.shape)