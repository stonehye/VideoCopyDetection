from multiprocessing import Pool
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import numpy as np
import faiss
import torch
import os
from typing import Union
import pickle
import glob


class Period(object):
    # half-closed form [a, b)
    def __init__(self, s, e):
        self.start, self.end = (s, e) if s < e else (e, s)

    def __repr__(self):
        return '{} - {}'.format(self.start, self.end)

    @property
    def length(self):
        return self.end - self.start

    def __add__(self, v: Union[int, float]):
        self.start += v
        self.end += v
        return self

    def __sub__(self, v: Union[int, float]):
        self.start -= v
        self.end -= v
        return self

    def __mul__(self, v: Union[int, float]):
        self.start *= v
        self.end *= v
        return self

    def is_overlap(self, o):
        assert isinstance(o, Period)
        return not ((self.end <= o.start) or (o.end <= self.start))

    def is_in(self, o):
        assert isinstance(o, Period)
        return o.start <= self.start and self.end <= o.end

    # self.start <= o.start <= o.end <= self.end
    def is_wrap(self, o):
        assert isinstance(o, Period)
        return self.start <= o.start and o.end <= self.end

    def intersection(self, o):
        assert isinstance(o, Period)
        return Period(max(self.start, o.start), min(self.end, o.end)) if self.is_overlap(o) else None

    # if not overlap -> self
    def union(self, o):
        assert isinstance(o, Period)
        return Period(min(self.start, o.start), max(self.end, o.end)) if self.is_overlap(o) else None

    def IOU(self, o):
        try:
            intersect = self.intersection(o)
            union = self.union(o)
            iou = intersect.length / union.length
        except:
            iou = 0
        return iou


class TN(object):
    def __init__(self, D, video_idx, frame_idx, TEMP_WND=10, MIN_LEN=5, THRESHOLD=-1):
        self.TEMP_WND = TEMP_WND
        self.MIN_LEN = MIN_LEN
        self.THRESHOLD = THRESHOLD

        # [# of query index, topk]
        self.video_index = video_idx
        self.frame_index = frame_idx
        self.dist = D

        self.query_length = D.shape[0]
        self.topk = D.shape[1]

        # dist, count, query start, reference start
        self.paths = np.empty((*D.shape, 4), dtype=object)

    def find_previous_linkable_nodes(self, t, r):
        video_idx, frame_idx = self.video_index[t, r], self.frame_index[t, r]
        min_prev_time = max(0, t - self.TEMP_WND)

        # find previous nodes that have (same video index) and (frame timestamp - wnd <= previous frame timestamp < frame timestamp)
        time, rank = np.where((self.dist[min_prev_time:t, ] >= self.THRESHOLD) &
                              (self.video_index[min_prev_time:t, ] == video_idx) &
                              (self.frame_index[min_prev_time:t, ] >= frame_idx - self.TEMP_WND) &
                              (self.frame_index[min_prev_time:t, ] < frame_idx)
                              )
        return np.stack((time + min_prev_time, rank), axis=-1)

    def fit(self):
        for time in range(self.query_length):
            for rank in range(self.topk):
                prev_linkable_nodes = self.find_previous_linkable_nodes(time, rank)

                if not len(prev_linkable_nodes):
                    self.paths[time, rank] = [self.dist[time, rank],
                                              1,
                                              time,
                                              self.frame_index[time, rank]]
                else:
                    # priority : count, path length, path score
                    prev_time, prev_rank = max(prev_linkable_nodes, key=lambda x: (
                        self.paths[x[0], x[1], 1],
                        self.frame_index[time, rank] -
                        self.paths[x[0], x[1], 3],
                        # self.paths[x[0], x[1], 0],
                    ))

                    prev_path = self.paths[prev_time, prev_rank]
                    self.paths[time, rank] = [prev_path[0] + self.dist[time, rank],
                                              prev_path[1] + 1,
                                              prev_path[2],
                                              prev_path[3]]

        # connect and filtering paths
        candidate = defaultdict(list)
        for time in reversed(range(self.query_length)):
            for rank in range(self.topk):
                score, count, q_start, r_start = self.paths[time, rank]
                if count >= self.MIN_LEN:
                    video_idx, frame_idx = self.video_index[time, rank], self.frame_index[time, rank]
                    q = Period(q_start, time + 1)
                    r = Period(r_start, frame_idx + 1)
                    path = (video_idx, q, r, score, count)
                    # remove include path
                    flag = True
                    for n, c in enumerate(candidate[video_idx]):
                        if path[1].is_wrap(c[1]) and path[2].is_wrap(c[2]):
                            candidate[video_idx][n] = path
                            flag = False
                            break
                        elif path[1].is_in(c[1]) and path[2].is_in(c[2]):
                            flag = False
                            break
                    if flag:
                        candidate[video_idx].append(path)

        # remove overlap path
        for video, path in candidate.items():
            candidate[video] = self.nms_path(path)

        candidate = [c for cc in candidate.values() for c in cc]
        return candidate

    def nms_path(self, path):
        l = len(path)
        path = np.array(sorted(path, key=lambda x: (x[4], x[3], x[2].length, x[1].length), reverse=True))

        keep = np.array([True] * l)
        overlap = np.vectorize(lambda x, a: x.is_overlap(a))
        for i in range(l - 1):
            if keep[i]:
                keep[i + 1:] = keep[i + 1:] & \
                               (~(overlap(path[i + 1:, 1], path[i, 1]) & overlap(path[i + 1:, 2], path[i, 2])))
        path = path.tolist()
        path = [path[n] for n in range(l) if keep[n]]

        return path


@torch.no_grad()
def extract_videos(model, loader):
    model.eval()
    videos = OrderedDict()
    length = OrderedDict()
    features = []
    bar = tqdm(loader, ncols=200, unit='batch')
    for i, (path, frame) in enumerate(loader):
        out = model(frame)
        features.append(out.cpu().numpy())
        bar.update()
        for p in path:
            vid = os.path.basename(os.path.dirname(p))
            length.setdefault(vid, 0)
            length[vid] += 1
            videos[vid] = vid
    bar.close()
    length = list(length.values())
    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)
    videos = {v: n for n, v in enumerate(videos)}

    return np.concatenate(features), videos, index


def load(path):
    feat = torch.load(path)
    return feat


def load_features(videos, feature_root):
    pool = Pool()
    bar = tqdm(videos, mininterval=1, ncols=150)
    features = [pool.apply_async(load, args=[os.path.join(feature_root, f'{v}.pth')], callback=lambda *a: bar.update())
                for v in videos]
    pool.close()
    pool.join()
    bar.close()
    features = [f.get() for f in features]
    length = [f.shape[0] for f in features]

    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)
    # index = np.transpose(np.vstack([start[:-1], start[1:]]))
    videos = {v: n for n, v in enumerate(videos)}
    return np.concatenate(features), videos, index


def scan_vcdb_annotation(root):
    def parse(ann):
        a, b, *times = ann.strip().split(',')
        times = [sum([60 ** (2 - n) * int(u) for n, u in enumerate(t.split(':'))]) for t in times]
        return [a, b, *times]

    groups = os.listdir(root)
    annotations = defaultdict(list)

    for g in groups:
        f = open(os.path.join(root, g), 'r')
        group = os.path.splitext(g)[0]
        for l in f.readlines():
            a, b, sa, ea, sb, eb = parse(l)
            annotations[a] += [[group, a, b, sa, ea, sb, eb]]
            if a != b:
                annotations[b] += [[group, b, a, sb, eb, sa, ea]]

    return annotations


# precision - detect path 중 gt와 1 frame 이상 겹치면 모두 정답
# recall - ground truth 중 매칭된 path가 있으면 찾은 것으로 봄
# SP=|correctly retrieved segments|/|all retrieved segments|
# SR=|correctly retrieved segments|/|groundtruth copy segments|. I
def match(path, gt):
    def vectorize_match(idx):
        y, x = divmod(idx, len(gt))
        if path[y][0] == gt[x][0] and path[y][1].is_overlap(gt[x][1]) and path[y][2].is_overlap(gt[x][2]):
            return 1  # x
        return 0

    d, g = 0, 0
    if len(path) and len(gt):
        correct = np.arange(0, len(path) * len(gt))
        ret = np.vectorize(vectorize_match)(correct).reshape(len(path), len(gt))
        d = np.count_nonzero(np.sum(ret, axis=1))
        g = np.count_nonzero(np.sum(ret, axis=0))

    return d, g


def idx2time(query, videos_namelist, candidates, db_intervals):
    new_candidates = []
    for can in candidates:
        reference = videos_namelist[can[0]]

        query_startidx = can[1].start
        query_endidx = can[1].end
        ref_startidx = can[2].start
        ref_endidx = can[2].end

        query_start = db_intervals[query][query_startidx][0]
        query_end = db_intervals[query][query_endidx][1]
        ref_start = db_intervals[reference][ref_startidx][0]
        ref_end = db_intervals[reference][ref_endidx][1]

        new_candidates += [[can[0], Period(round(query_start), round(query_end)), Period(round(ref_start), round(ref_end)), can[3], can[4]]]

    return new_candidates


def test(topk, tn_param):
    vcdb_core_video = np.load('/nfs_shared/MLVD/VCDB/meta/vcdb_videos_core.npy')
    annotation = scan_vcdb_annotation('/nfs_shared_/hkseok/VCDB/videos/annotation')
    feature_path = '/nfs_shared_/hkseok/BOW/multiple/vcdb_core-1fps-MobileNet_triplet_sum-5sec'

    feature, videos, loc = load_features(vcdb_core_video, feature_path)

    table = dict()
    count = 0
    for video_idx, ran in enumerate(loc):
        for features_idx in range(ran[1] - ran[0]):
            table[count] = (video_idx, features_idx)
            count += 1
    mapping = np.vectorize(lambda x, table: table[x])

    index = faiss.IndexFlatIP(feature.shape[1])
    index = faiss.index_cpu_to_all_gpus(index)
    faiss.normalize_L2(feature)
    index.add(feature)
    # search_dst = '/nfs_shared_/hkseok/search/local/multiple/vcdb_core-1fps-MobileNet_local-5sec-sum'

    a, b, c, d = 0, 0, 0, 0
    for n, (query, gt) in enumerate(annotation.items(), start=1):
        # D = np.load(os.path.join(search_dst, query + '_D.npy'))[:, :topk]
        # I = np.load(os.path.join(search_dst, query + '_I.npy'))[:, :topk]

        q_id = videos[query]
        start, end = loc[q_id]
        q_feat = feature[start:end]
        D, I = index.search(q_feat, topk)

        idx = mapping(I, table)
        vidx, fidx = idx[0], idx[1]  # video idx, frame idx

        tn = TN(D, vidx, fidx, *tn_param)
        candidate = tn.fit()
        candidate = [[x[0], Period(x[1].start * 5, x[1].end * 5), Period(x[2].start * 5, x[2].end * 5), x[3], x[4]] for
                     x in candidate]
        ground = [(videos[g[2]], Period(g[3], g[4]), Period(g[5], g[6])) for g in gt]

        aa, cc = match(candidate, ground)
        bb, dd = len(candidate), len(ground)

        a += aa  # correct candidate
        b += bb  # all candidate
        c += cc  # correct ground
        d += dd  # all ground
        pp = aa / (bb + 1e-12)  # precision
        rr = cc / (dd + 1e-12)  # recall
        ff = 2 * pp * rr / (pp + rr + 1e-12)  # f1-score

        p = a / (b + 1e-12)
        r = c / (d + 1e-12)
        f = 2 * p * r / (p + r + 1e-12)

        # print(n, '======')
        # print('detect', len(candidate), sorted(candidate, key=lambda x: (x[0], -x[4], x[3])))
        # print('gt', len(ground), sorted(ground, key=lambda x: x[0]))
        # print(f'{n}: {f:.4f} {p:.4f} {r:.4f} ({ff:.4f} {pp:.4f} {rr:.4f}) {a:>5d}({aa:>3d}) {b:>5d}({bb:>3d}) {c:>5d}({cc:>3d}) {d:>5d}({dd:>3d})')

    p = a / (b + 1e-12)
    r = c / (d + 1e-12)
    f = 2 * p * r / (p + r + 1e-12)
    print(f'{topk} {tn_param[0]} {tn_param[1]} {tn_param[2]} {a:>5d} {b:>5d} {c:>5d} {d:>5d} {p:.4f} {r:.4f} {f:.4f} ')


if __name__ == '__main__':
    for topk in [200,250,300]:
        for temp_wnd in [50, 100, 150,200]:
            for min_length in [3]:
                for thr in [0.3,0.35]:
                    test(topk=topk, tn_param=[temp_wnd, min_length, thr])