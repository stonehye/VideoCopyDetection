from module.Temporal_Network import *
from pipeline import *
from torchvision import transforms as trn
from typing import Union
import numpy as np
import faiss
import os
import pickle
import datetime
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


def is_overlap(detect, ground):
    detect_start = detect[0]
    detect_end = detect[1]
    ground_start = sum(int(x) * 60 ** i for i, x in enumerate(reversed(ground[0].split(':'))))
    ground_end = sum(int(x) * 60 ** i for i, x in enumerate(reversed(ground[1].split(':'))))


    # detect_start = datetime.datetime.strptime(detect[0], "%H:%M:%S")
    # detect_end = datetime.datetime.strptime(detect[1], "%H:%M:%S")
    # ground_start = datetime.datetime.strptime(ground[0], "%H:%M:%S")
    # ground_end = datetime.datetime.strptime(ground[1], "%H:%M:%S")

    return not (detect_end < ground_start or ground_end < detect_start)


decode_size = 256
group_count = 32
# cnn_model = MobileNet_RMAC().cuda()
# cnn_model = nn.DataParallel(cnn_model)
aggr_model = Segment_Maxpooling()
transform = trn.Compose([
    trn.Resize((224, 224)),
    trn.ToTensor(),
    trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

basename = '/nfs_shared_/hkseok/local/' + 'vcdb_core-0-mobilenet_avg-32-segment_maxpooling'
db_features = np.load(basename + '_feature.npy')
db_index = np.load(basename + '_index.npy')
db_paths = np.load(basename + '_paths.npy')
db_video_dict={os.path.basename(v).replace('.pth',''):k for k,v in enumerate(db_paths)}

with open(basename + '_intervals.pkl', 'rb') as fr:
    db_intervals = pickle.load(fr)
# print(db_intervals)

with open('/nfs_shared_/hkseok/VCDB/annotation.pkl', 'rb') as fr:
    ground_truth = pickle.load(fr)

table=dict(); count = 0
for video_idx, ran in enumerate(db_index):
    for features_idx in range(ran[1]-ran[0]):
        table[count] = (video_idx, features_idx)
        count+=1
mapping = np.vectorize(lambda x, table: table[x])

result = {'detect': {'hit': 0, 'miss': 0}, 'ground': {'hit': [], 'miss': 0}}
cnt = 0
a, b, c, d = 0, 0, 0, 0
d = len(ground_truth)
for video in glob.glob('/nfs_shared_/hkseok/VCDB/videos/core/*'):
    print("{} / {}".format(cnt+1, len(glob.glob('/nfs_shared_/hkseok/VCDB/videos/core/*'))))
    cnt += 1

    query_path = video
    query_video = os.path.basename(query_path)
    # query_video_features, shots = extract_segment_fingerprint(query_path, decode_size, transform, cnn_model, aggr_model, group_count, 'local')
    query_video_features = torch.load(os.path.join(basename, query_video + '.pth'))
    query_video_features = query_video_features.numpy()

    # search top k features per each query frames
    l2index = faiss.IndexFlatL2(db_features.shape[1])
    l2index.add(db_features)
    D, I = l2index.search(query_video_features, k=100)

    I_to_frame_index = np.dstack(mapping(I, table))  # index to (video id , frame id)
    # print(I_to_frame_index[:,:,0]) # temporal network rank
    # print(I_to_frame_index[:,:,1]) # temporal network path

    # find copy segment
    temporal_network = TN(D, I_to_frame_index, 1, 2)
    candidate = temporal_network.fit()

    # candidate=[(c[0],
    #             Period(*c[1]),
    #             Period(*c[2]),
    #             c[3],
    #             c[4]) for c in candidate]

    # ground = []
    # for g in ground_truth:
    #     if g.get(query_video) is not None:
    #         keys = list(g.keys())
    #         if len(keys) == 1: keys.append(keys[0])
    #         ref_video = None
    #         if keys[0] == query_video: ref_video= keys[1]
    #         else: ref_video= keys[0]
    #
    #         query_start = sum(int(x) * 60 ** i for i, x in enumerate(reversed(g[query_video][0].split(':'))))
    #         query_end = sum(int(x) * 60 ** i for i, x in enumerate(reversed(g[query_video][1].split(':'))))
    #         ref_start = sum(int(x) * 60 ** i for i, x in enumerate(reversed(g[ref_video][0].split(':'))))
    #         ref_end = sum(int(x) * 60 ** i for i, x in enumerate(reversed(g[ref_video][1].split(':'))))
    #
    #         ground += [(db_video_dict[ref_video], Period(query_start, query_end), Period(ref_start, ref_end))]
    # aa, cc = match(candidate, ground)
    # bb, dd = len(candidate), len(ground)
    #
    # a += aa
    # b += bb
    # c += cc
    # d += dd
    # pp = aa / (bb + 1e-12)
    # rr = cc / (dd + 1e-12)
    # ff = 2 * pp * rr / (pp + rr + 1e-12)
    #
    # p = a / (b + 1e-12)
    # r = c / (d + 1e-12)
    # f = 2 * p * r / (p + r + 1e-12)
    #
    # cnt += 1
    # print(cnt, '======')
    # print('detect', len(candidate), sorted(candidate, key=lambda x: (x[0], -x[4], x[3])))
    # print('gt', len(ground), sorted(ground, key=lambda x: x[0]))
    # print(
    #     f'{cnt}: {f:.4f} {p:.4f} {r:.4f} ({ff:.4f} {pp:.4f} {rr:.4f}) {a:>5d}({aa:>3d}) {b:>5d}({bb:>3d}) {c:>5d}({cc:>3d}) {d:>5d}({dd:>3d})')



    query_video = os.path.basename(query_path)
    print(query_video)

    ground = [gt for gt in ground_truth if query_video in list(gt.keys())]

    b += len(candidate)
    for can in candidate:
        ref_video = os.path.basename(db_paths[can[0]]).replace('.pth', '')

        query_start = db_intervals[query_video][can[1][0]][0]
        query_end = db_intervals[query_video][can[1][1]][1]
        ref_start = db_intervals[ref_video][can[2][0]][0]
        ref_end = db_intervals[ref_video][can[2][1]][1]

        candidate_ground = []
        for gt in ground:
            keys = list(gt.keys())
            if len(keys) == 1: keys.append(keys[0])
            if [query_video, ref_video] == keys or [ref_video, query_video] == keys:
                candidate_ground.append(gt)

        for gt in candidate_ground:
            if is_overlap([ref_start, ref_end],gt[ref_video]) and is_overlap([query_start, query_end],gt[query_video]):
                result['ground']['hit'].append(gt)
                # c = len([g for g in ground_truth if g in result['ground']['hit']])
                a += 1
    # print(a,b,c,d)

c = len([g for g in ground_truth if g in result['ground']['hit']])

precision = a/b
recall = c/d
print(a,b,c,d, a/b, c/d, 2*precision*recall/(precision+recall))


