import os
import numpy as np
from multiprocessing import Pool
import torch
import tqdm
import faiss
from collections import defaultdict
import pickle as pk


def scan_vcdb_annotation(root):
    def parse(ann):
        a, b, *times = ann.strip().split(',')
        times = [sum([60 ** (2 - n) * int(u) for n, u in enumerate(t.split(':'))]) for t in times]
        return [a, b, *times]

    groups = os.listdir(root)
    annotations = []

    for g in groups:
        f = open(os.path.join(root, g), 'r')
        annotations += [[os.path.splitext(g)[0], *parse(l)] for l in f.readlines()]

    return annotations


def load(path):
    feat = torch.load(path)
    return feat


def load_features(paths):
    pool = Pool()
    bar = tqdm.tqdm(range(len(paths)), mininterval=1, ncols=150)
    features = [pool.apply_async(load, args=[path], callback=lambda *a: bar.update()) for path in paths]
    pool.close()
    pool.join()
    bar.close()
    features = [f.get() for f in features]
    length = [f.shape[0] for f in features]

    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)

    return np.concatenate(features), np.array(length), index


if __name__ == '__main__':
    np.set_printoptions(threshold=3, edgeitems=3)
    vcdb_videos = np.load('/nfs_shared/MLVD/VCDB/meta/vcdb_videos_core.npy')
    segment_annotation = scan_vcdb_annotation('/nfs_shared/MLVD/VCDB/annotation')
    video2id = {v: n for n, v in enumerate(vcdb_videos)}
    feature_base = '/nfs_shared_/hkseok/vcdb_core-1-mobilenet_avg-5-segment_maxpooling' # global feature
    feature_base2 = '/nfs_shared_/hkseok/BOW/multiple/vcdb_core-1fps-MobileNet_local-5sec-sum' # local feature

    name='multiple-vcdb_core-1fps-MobileNet-5sec-weightedsum3'

    feature_path = np.char.add(np.char.add(feature_base+'/', vcdb_videos), '.pth')
    feature, length, location = load_features(feature_path)

    feature_path2 = np.char.add(np.char.add(feature_base2 + '/', vcdb_videos), '.pth')
    feature2, length2, location2 = load_features(feature_path2)

    db_interval = dict()
    for n, v in enumerate(vcdb_videos):
        vid = video2id[v]
        db_interval[v] = [[i * 5, (i + 1) * 5] for i in range(0, length[vid])]

    feature_annotation=defaultdict(list)
    for ann in segment_annotation:
        g, a, b, sa, ea, sb, eb = ann
        ai = [n for n, i in enumerate(db_interval[a]) if not (i[1] <= sa or ea <= i[0])]
        bi = [n for n, i in enumerate(db_interval[b]) if not (i[1] <= sb or eb <= i[0])]

        cnt=len(ai)
        af = np.linspace(ai[0], ai[-1], cnt, endpoint=True, dtype=np.int).reshape(-1, 1)
        bf = np.linspace(bi[0], bi[-1], cnt, endpoint=True, dtype=np.int).reshape(-1, 1)
        feature_annotation[a].append([b, np.concatenate([af, bf], axis=1)])
        if a!=b:
            cnt = len(bi)
            af = np.linspace(ai[0], ai[-1], cnt, endpoint=True, dtype=np.int).reshape(-1, 1)
            bf = np.linspace(bi[0], bi[-1], cnt, endpoint=True, dtype=np.int).reshape(-1, 1)
            feature_annotation[b].append([a, np.concatenate([bf, af], axis=1)])

    index = faiss.IndexFlatIP(feature.shape[1])
    index.add(feature)

    index2 = faiss.IndexFlatIP(feature2.shape[1])
    faiss.normalize_L2(feature2)
    index2.add(feature2)

    result = dict()
    for qv_idx, qv in enumerate(vcdb_videos):
        ann = feature_annotation[qv]
        qv_feature = feature[location[qv_idx][0]:location[qv_idx][1]]
        qv_feature2 = feature2[location[qv_idx][0]:location[qv_idx][1]]

        print(qv_idx, qv, length[qv_idx])

        D, I = index.search(qv_feature, feature.shape[0])
        D2, I2 = index2.search(qv_feature2, feature2.shape[0])

        new_D = []
        new_I = []
        for i in range(length[qv_idx]):
            global_feature = np.array((D[i], I[i])).T
            local_feature = np.array((D2[i], I2[i])).T
            global_feature = global_feature[global_feature[:,1].argsort()]
            local_feature = local_feature[local_feature[:, 1].argsort()]
            summed_feature = np.array((0.5*global_feature[:,0]+ 0.5*local_feature[:, 0], global_feature[:,1])).T
            summed_feature = summed_feature[summed_feature[:, 0].argsort()][::-1]
            new_D.append(summed_feature[:,0])
            new_I.append(summed_feature[:,1])
        new_I = np.array(new_I)
        new_D = np.array(new_D)

        result[qv] = defaultdict(list)
        for a in ann:
            loc = location[video2id[a[0]]]
            query_time = a[1][:, 0]
            ref_idx = a[1][:, 1] + loc[0]
            rank = [np.where(np.abs(new_I[t, :] - ref_idx[n]) <= 2)[0][0] for n, t in enumerate(query_time)]
            ret = np.vstack([a[1][:, 0], a[1][:, 1], rank]).T
            result[qv][a[0]].append(ret)

    pk.dump(result, open(f'{name}.pkl', 'wb'))
    result = pk.load(open(f'{name}.pkl', 'rb'))
    result_per_feature = dict()
    for qv, ret in result.items():
        result_per_feature[qv] = defaultdict(list)
        for rf, ranks in ret.items():
            for r in ranks:
                for i in r:
                    result_per_feature[qv][i[0]].append(i[2])
    print(result_per_feature)