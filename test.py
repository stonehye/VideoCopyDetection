from module.Temporal_Network import *
from pipeline import *
from torchvision import transforms as trn
import numpy as np
import faiss
import os

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
query_video_features, shots = extract_segment_fingerprint(query_path, decode_size, transform, cnn_model, aggr_model, group_count, 'local')
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
