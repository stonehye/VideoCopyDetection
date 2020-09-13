from pipeline import *
from torchvision import transforms as trn
import glob
import os
import pickle


dst = '/nfs_shared/hkseok/'
basename = 'vcdb_core-2-mobilenet_avg-4-segment_maxpooling'
# {dataset}-{decode_rate}-{cnn_extractor}-{group_count}-{aggr_model}

decode_rate = 2
decode_size = 256
group_count = 4
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

""" Video segment feature """
empty_shotlist_count = 0 # Number of videos without shot boundary detection
video_list = glob.glob('/nfs_shared/MLVD/VCDB/videos/*') # reference videos path
for video in video_list:
    segment_fingerprint = extract_segment_fingerprint(video, decode_rate, decode_size, transform, cnn_model, aggr_model, group_count)
    videoname = os.path.basename(video)
    dst_path = os.path.join(pth_dir, videoname + '.pth')
    torch.save(segment_fingerprint, dst_path)
print("Num of videos without boundary detection: {}".format(empty_shotlist_count))

""" feature DB """
db_feature, db_length, db_index, db_paths = load_segment_fingerprint(pth_dir)
with open(os.path.join(dst, basename + '_feature.pkl')) as fw:
    pickle.dump(db_feature, fw)
with open(os.path.join(dst, basename + '_length.pkl')) as fw:
    pickle.dump(db_length, fw)
with open(os.path.join(dst, basename + '_index.pkl')) as fw:
    pickle.dump(db_index, fw)
with open(os.path.join(dst, basename + '_paths.pkl')) as fw:
    pickle.dump(db_paths, fw)