from pipeline import *
from torchvision import transforms as trn
import glob
import os
import pickle
import time


dst = '/nfs_shared_/hkseok/'
basename = 'vcdb_core-0-mobilenet_avg-32-segment_maxpooling'
# {dataset}-{decode_rate}-{cnn_extractor}-{group_count}-{aggr_model}

decode_size = 256
group_count = 32
cnn_model = MobileNet_AVG().cuda()
cnn_model = nn.DataParallel(cnn_model)
aggr_model = Segment_Maxpooling()
transform = trn.Compose([
    trn.Resize((224, 224)),
    trn.ToTensor(),
    trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

pth_dir = os.path.join(dst, basename) # pth file path
if not os.path.isdir(pth_dir):
    os.makedirs(pth_dir)


""" video segment feature """
start = time.time()
video_list = glob.glob('/nfs_shared_/hkseok/VCDB/videos/core/*')
empty_shotlist_count = 0 # Number of videos without shot boundary detection
feature_intervals = {}
for idx, video in enumerate(video_list):
    videoname = os.path.basename(video)
    print(videoname, idx)
    segment_fingerprint, shots = extract_segment_fingerprint(video, decode_size, transform, cnn_model, aggr_model, group_count, 'minmax')
    dst_path = os.path.join(pth_dir, videoname + '.pth')
    torch.save(segment_fingerprint, dst_path)
    feature_intervals[videoname] = shots
print("Num of videos without boundary detection: {}".format(empty_shotlist_count))
end = time.time() - start
print("{} video segment feature extraction times: {}sec".format(len(video_list), end))
print("Average feature extraction time: {}sec".format(end/len(video_list)))


""" feature DB """
db_feature, db_length, db_index, db_paths = load_segment_fingerprint(pth_dir)
np.save(os.path.join(dst, basename + '_feature.npy'), db_feature)
np.save(os.path.join(dst, basename + '_length.npy'), db_length)
np.save(os.path.join(dst, basename + '_index.npy'), db_index)
np.save(os.path.join(dst, basename + '_paths.npy'), db_paths)

dst_path = os.path.join(dst, basename + '_intervals.pkl')
with open(dst_path, "wb") as fw:
    pickle.dump(feature_intervals, fw)

print(feature_intervals)
