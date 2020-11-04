import os
import numpy as np
from multiprocessing import Pool
import torch
import tqdm
import faiss
from collections import defaultdict
import pickle as pk
from pprint import pprint


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

    # frame_annotations = defaultdict(list)
    # for ann in annotations:
    #     g, a, b, sa, ea, sb, eb = ann
    #     cnt = min(ea - sa, eb - sb)
    #     af = np.linspace(sa, ea, cnt, endpoint=False, dtype=np.int).reshape(-1, 1)
    #     bf = np.linspace(sb, eb, cnt, endpoint=False, dtype=np.int).reshape(-1, 1)
    #     frame_annotations[a].append([b, np.concatenate([af, bf], axis=1)])
    #     if a != b:
    #         frame_annotations[b].append([a, np.concatenate([bf, af], axis=1)])
    # return annotations, frame_annotations
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
    # segment_annotation, feature_annotation = scan_vcdb_annotation('/MLVD/VCDB/annotation')
    segment_annotation = scan_vcdb_annotation('/nfs_shared/MLVD/VCDB/annotation')
    video2id = {v: n for n, v in enumerate(vcdb_videos)}
    # feature_base = '/nfs_shared/MLVD/VCDB/vcdb_core-5sec-5frame-mobilenet-avg-max/ep0/'
    # feature_base= '/nfs_shared/ms/hkseok_tmp/minmax/vcdb_core_minmax_545-0-mobilenet_avg-10-segment_maxpooling'
    # feature_base = '/nfs_shared/ms/hkseok_tmp/local/vcdb_core-0-mobilenet_avg-32-segment_maxpooling'
    # feature_base = '/nfs_shared_/hkseok/BOW/vcdb_core_900-1fps-mobilenet-5sec'
    # feature_base = '/nfs_shared/MLVD/VCDB/vcdb_core-1sec-1frame-mobilenet-avg/ep0/'
    feature_base = '/nfs_shared_/hkseok/BOW/vcdb_core-1fps-mobilenet-5sec-400size'

    name='fourth2'

    feature_path = np.char.add(np.char.add(feature_base+'/', vcdb_videos), '.pth')
    feature, length, location = load_features(feature_path)
    # print(feature_annotation)

    table = dict()
    count = 0
    for video_idx, ran in enumerate(location):
        for features_idx in range(ran[1] - ran[0]):
            table[count] = (video_idx, features_idx)
            count += 1
    # table = {i: (n, i - l[0]) for n, l in enumerate(location) for i in range(l[0], l[1])}
    mapping = np.vectorize(lambda x, table: table[x])
    print(table)

    db_interval = dict()
    for n, v in enumerate(vcdb_videos):
        vid = video2id[v]
        db_interval[v] = [[i * 5, (i + 1) * 5] for i in range(0, length[vid])]
        # print(length[vid], location[vid],len(db_interval[v]), db_interval[v])

    intv=0
    c=0
    for k,vv in db_interval.items():
        for v in vv:
            intv+=(v[1]-v[0])
            c+=1
        # db_interval[k]=[[vv[n][0],vv[n+1][0]] for n,v in enumerate(vv[:-1])]+[vv[-1]]
    print(intv,c,intv/c)


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

    result = dict()
    for qv_idx, qv in enumerate(vcdb_videos):
        ann = feature_annotation[qv]
        qv_feature = feature[location[qv_idx][0]:location[qv_idx][1]]
        print(qv_idx, qv, qv_feature.shape, length[qv_idx])
        D, I = index.search(qv_feature, feature.shape[0])
        result[qv] = defaultdict(list)
        for a in ann:
            loc = location[video2id[a[0]]]
            query_time = a[1][:, 0]
            ref_idx = a[1][:, 1] + loc[0]
            rank = [np.where(np.abs(I[t, :] - ref_idx[n]) <= 2)[0][0] for n, t in enumerate(query_time)]
            ret = np.vstack([a[1][:, 0], a[1][:, 1], rank]).T
            result[qv][a[0]].append(ret)

    # print([ for k,v in result['00274a923e13506819bd273c694d10cfa07ce1ec.flv'] for vv in v])


    pk.dump(result, open(f'{name}.pkl', 'wb'))
    result = pk.load(open(f'{name}.pkl', 'rb'))
    # print(result)
    result_per_feature = dict()
    for qv, ret in result.items():
        result_per_feature[qv] = defaultdict(list)
        for rf, ranks in ret.items():
            for r in ranks:
                for i in r:
                    result_per_feature[qv][i[0]].append(i[2])
    print(result_per_feature)