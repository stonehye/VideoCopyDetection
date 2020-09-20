from module.Temporal_Network import *
from pipeline import *
from torchvision import transforms as trn
import numpy as np
import faiss
import os

from datetime import datetime, timedelta, timezone
from dataset.VCDB import VCDB, NestedListDataset
from utils.utils import *
from nets.summary import summary
import logging

kst = timezone(timedelta(hours=9))


decode_rate = 2
decode_size = 256
group_count = 5
cnn_model = MobileNet_AVG().cuda()
cnn_model = nn.DataParallel(cnn_model)
aggr_model = Segment_Maxpooling()
transform = trn.Compose([
    trn.Resize((224, 224)),
    trn.ToTensor(),
    trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

db_features = np.load('/nfs_shared_/hkseok/vcdb_core-0-mobilenet_avg-5-segment_maxpooling_feature.npy')
db_index = np.load('/nfs_shared_/hkseok/vcdb_core-0-mobilenet_avg-5-segment_maxpooling_index.npy')
db_paths = np.load('/nfs_shared_/hkseok/vcdb_core-0-mobilenet_avg-5-segment_maxpooling_paths.npy')

table=dict(); count = 0
for video_idx, ran in enumerate(db_index):
    for features_idx in range(ran[1]-ran[0]):
        table[count] = (video_idx, features_idx)
        count+=1
mapping = np.vectorize(lambda x, table: table[x])

query_path = '/nfs_shared/MLVD/VCDB/videos/5b46e9007b0add1d73a11d2f4414efe35b017acb.flv'
query_video_features, shots = extract_segment_fingerprint(query_path, decode_size, transform, cnn_model, aggr_model, group_count, 'minmax')
query_video_features = query_video_features.numpy()

# search top k features per each query frames
l2index = faiss.IndexFlatL2(db_features.shape[1])
l2index.add(db_features)
D, I = l2index.search(query_video_features, k=50)

I_to_frame_index = np.dstack(mapping(I, table))  # index to (video id , frame id)
# print(I_to_frame_index[:,:,0]) # temporal network rank
# print(I_to_frame_index[:,:,1]) # temporal network path

# find copy segment
temporal_network = TN(D, I_to_frame_index, 3, 3)
candidate = temporal_network.fit()
# [(video_id,[query],[reference],dist,count) ... ]
print(candidate)

print("query video: {}".format(os.path.basename(query_path)))
for can in candidate:
    print("{}th video: {}".format(can[0], db_paths[can[0]]))


def main():
    start_time = datetime.now(tz=kst).strftime("%Y-%m-%d %H:%M:%S")

    # VCDB
    nfold = 1
    valid_idx = 0
    segment_length = 5
    frames_per_segment = 1
    reference_cnt = None # Reference Video Count. None: not use background
    backgroud = True if reference_cnt else False

    # model
    ckpt = None # checkpoint dir

    # Dataloader
    batch = 32
    worker = 4

    # Temporal Network
    topk = 50
    score_thr = 0.7
    temp_wnd = 5
    min_match = 5

    log = 'out.log'
    logger, log = init_logger(log)

    vcdb = VCDB(root='/nfs_shared_/hkseok/VCDB/videos/core')
    vcdb.set_vcdb_parameters(nfold=nfold, valid_idx=valid_idx,
                             segment_length=segment_length, frames_per_segment=frames_per_segment)
    logger.info(vcdb.toStr())

    ref_videos, core_cnt = vcdb.get_reference_videos(background=backgroud, total=reference_cnt)
    eval_loader = DataLoader(NestedListDataset([]), batch_size=batch, shuffle=False, num_workers=worker)

    model = MobileNet_AVG().cuda()
    logger.info('Model : {}'.format(model))
    logger.info('Model Summary')
    logger.info(summary(model, (frames_per_segment, 3, 224, 224)))



