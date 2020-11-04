from module.Temporal_Network import *
from pipeline import *
from torchvision import transforms as trn
import numpy as np
import faiss
import os
import pickle
import datetime
import glob


def is_overlap(detect, ground):
    detect_start = datetime.datetime.strptime(detect[0], "%H:%M:%S")
    detect_end = datetime.datetime.strptime(detect[1], "%H:%M:%S")
    ground_start = datetime.datetime.strptime(ground[0], "%H:%M:%S")
    ground_end = datetime.datetime.strptime(ground[1], "%H:%M:%S")

    return not (detect_end < ground_start or ground_end < detect_start)


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
# cnt = 0
for video in glob.glob('/nfs_shared_/hkseok/VCDB/videos/core/*'):
    # print("{} / {}".format(cnt+1, len(glob.glob('/nfs_shared_/hkseok/VCDB/videos/core/*'))))
    # cnt += 1

    query_path = video
    query_video = os.path.basename(query_path)
    # query_video_features, shots = extract_segment_fingerprint(query_path, decode_size, transform, cnn_model, aggr_model, group_count, 'local')
    query_video_features = torch.load(os.path.join(basename, query_video + '.pth'))
    query_video_features = query_video_features.numpy()

    # search top k features per each query frames
    l2index = faiss.IndexFlatL2(db_features.shape[1])
    l2index.add(db_features)
    D, I = l2index.search(query_video_features, k=20)

    I_to_frame_index = np.dstack(mapping(I, table))  # index to (video id , frame id)
    # print(I_to_frame_index[:,:,0]) # temporal network rank
    # print(I_to_frame_index[:,:,1]) # temporal network path

    # find copy segment
    temporal_network = TN(D, I_to_frame_index, 3, 3)
    candidate = temporal_network.fit()
    # [(video_id,[query],[reference],dist,count) ... ]
    # print(candidate)
    # print(len(candidate), end=' ')

    query_video = os.path.basename(query_path)
    for can in candidate:
        ref_video = os.path.basename(db_paths[can[0]]).replace('.pth', '')
        query_start = str(datetime.timedelta(seconds=round(db_intervals[query_video][can[1][0]][0])))
        query_end = str(datetime.timedelta(seconds=round(db_intervals[query_video][can[1][1]][1])))
        ref_start = str(datetime.timedelta(seconds=round(db_intervals[ref_video][can[2][0]][0])))
        ref_end = str(datetime.timedelta(seconds=round(db_intervals[ref_video][can[2][1]][1])))

        # print(ref_video, query_start, query_end, ref_start, ref_end)
        for gt in ground_truth:
            isHit = False
            keys = list(gt.keys())
            if len(keys) ==1: keys.append(keys[0])
            if [query_video, ref_video] == keys or [ref_video, query_video] == keys:
                if is_overlap([ref_start, ref_end],gt[ref_video]) and is_overlap([query_start, query_end],gt[query_video]):
                    result['detect']['hit']+=1
                    result['ground']['hit'].append(gt)
                    isHit = True
                    print("{} and {} || gt: {}, {} || can: {}, {}".format(query_video, ref_video, gt[query_video], gt[ref_video], [query_start, query_end], [ref_start, ref_end]))
                if not isHit:
                    result['detect']['miss']+=1
    # print(result['detect'], result['ground']['miss'])
result['ground']['miss'] = len([g for g in ground_truth if g not in result['ground']['hit']])

TP = result['detect']['hit']
FP = result['detect']['miss']
FN = result['ground']['miss']
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
print("TP:{}, FP:{}, FN:{}".format(TP, FP, FN))
print("Precision: {}, Recall: {}, f1-score: {}".format(Precision, Recall, 2*Precision*Recall/(Precision+Recall)))

