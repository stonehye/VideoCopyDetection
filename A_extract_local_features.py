from pymediainfo import MediaInfo
import subprocess

from torchvision import transforms as trn
from torchvision.transforms.functional import resize
from PIL import Image

from nets.models import *
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from multiprocessing import Pool
import os

import glob
import argparse


class ListDataset(Dataset):
    def __init__(self, l, transform=None):
        self.l = l
        default_transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform = transform or default_transform

    def __getitem__(self, idx):
        return self.transform(self.l[idx])

    def __len__(self):
        return len(self.l)

    def __repr__(self):
        fmt_str = f'{self.__class__.__name__}\n'
        fmt_str += f'\tNumber of images : {self.__len__()}\n'
        trn_str = self.transform.__repr__().replace('\n', '\n\t')
        fmt_str += f"\tTransform : \n\t{trn_str}"

        return fmt_str


def parse_metadata(path):
    media_info = MediaInfo.parse(path)
    meta = {'file_path': path}
    for track in media_info.tracks:
        if track.track_type == 'General':
            meta['file_name'] = track.file_name + '.' + track.file_extension
            meta['file_extension'] = track.file_extension
            meta['format'] = track.format
            # meta['duration'] = float(track.duration)
            # meta['frame_count'] = int(track.frame_count)
            # meta['frame_rate'] = float(track.frame_rate)
        elif track.track_type == 'Video':
            meta['width'] = int(track.width)
            meta['height'] = int(track.height)
            meta['rotation'] = float(track.rotation or 0.)
            meta['codec'] = track.codec
    return meta


def decode_frames(video, meta, decode_rate, size):
    frames = []
    w, h = (meta['width'], meta['height']) if meta['rotation'] not in [90, 270] else (meta['height'], meta['width'])
    command = ['ffmpeg',
               '-hide_banner', '-loglevel', 'panic',
               '-vsync', '2',
               '-i', video,
               '-pix_fmt', 'bgr24',  # color space
               '-r', str(decode_rate),
               '-q:v', '0',
               '-vcodec', 'rawvideo',  # origin video
               '-f', 'image2pipe',  # output format : image to pipe
               'pipe:1']
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=w * h * 3)
    while True:
        raw_image = pipe.stdout.read(w * h * 3)
        pipe.stdout.flush()
        try:
            image = Image.frombuffer('RGB', (w, h), raw_image, "raw", 'BGR', 0, 1)
        except ValueError as e:
            break

        if size:
            image = resize(image, size)
        frames.append(image)
    return frames


@torch.no_grad()
def extract_frame_fingerprint(model, loader):
    model.eval()
    frame_fingerprints = []

    for im in loader:
        feature = model(im)
        frame_fingerprints.append(feature.cpu())
    frame_fingerprints = torch.cat(frame_fingerprints)

    return frame_fingerprints


def extract_segment_fingerprint(video, decode_rate, decode_size, transform, cnn_model,group_count, aggr_model):
    # parse video metadata
    meta = parse_metadata(video)
    print(meta)

    # decode all frames
    frames = decode_frames(video, meta, decode_rate, decode_size)
    print(len(frames))

    # extract frame fingerprint
    cnn_loader = DataLoader(ListDataset(frames, transform=transform), batch_size=64, shuffle=False, num_workers=4)
    frame_fingerprints = extract_frame_fingerprint(cnn_model, cnn_loader)
    print("extract frame fingerprint: ", frame_fingerprints.shape)

    if group_count != 1:
        k = group_count - frame_fingerprints.shape[0] % group_count
        if k != group_count:
            frame_fingerprints = torch.cat([frame_fingerprints, frame_fingerprints[-1:, ].repeat((k, 1, 1))])
        frame_fingerprints = frame_fingerprints.permute(0, 2, 1)
        frame_fingerprints = frame_fingerprints.reshape(-1, group_count * frame_fingerprints.shape[1], frame_fingerprints.shape[-1])
        frame_fingerprints = frame_fingerprints.permute(0, 2, 1)
        print("groupping: ", frame_fingerprints.shape)

    if aggr_model:
        ## multiple keyframe - segment local maxpooling
        frame_fingerprints = aggr_model(frame_fingerprints)
        print("aggregating segment feature: ", frame_fingerprints.shape)

    local_features = []
    local_features_set = torch.split(frame_fingerprints, 1)
    for set in local_features_set:
        temp = torch.split(set, 1, dim=2)
        temp = [t.squeeze(-1) for t in temp]
        local_features.append(temp)

    print(f"# of Segments: {len(local_features)}")
    print(f"# of localfeatures per segment: {len(local_features[0])}")
    print(f"shape of localfeature: {local_features[0][0].shape}")
    print("-" * 53)

    # if group_count != 1:
    #     # grouping fingerprints for each segment => If frame_fingerprints cannot be divided by group_count, the last is copied.
    #     k = group_count - frame_fingerprints.shape[0] % group_count
    #     if k != group_count:
    #         frame_fingerprints = torch.cat([frame_fingerprints, frame_fingerprints[-1:, ].repeat((k, 1, 1))])
    #     print("groupping: ", frame_fingerprints.shape)
    #
    # groupping_fingerprint = frame_fingerprints
    # groupping_fingerprint = torch.split(groupping_fingerprint, 5, dim=0)
    # variance_fingerprint = [f.var(dim=0).unsqueeze(0) for f in groupping_fingerprint]
    #
    # if not aggr_model:
    #     frame_fingerprints = frame_fingerprints.reshape(-1, group_count * frame_fingerprints.shape[1], frame_fingerprints.shape[-1])
    #     print("Concatenate fingerprint for each segment: ", frame_fingerprints.shape)
    # else:
    #     frame_fingerprints = aggr_model(frame_fingerprints)
    #     print("aggregating segment feature: ", frame_fingerprints.shape)
    #
    # local_features_set = torch.split(frame_fingerprints, 1)
    # concatenated_fingerprint = [torch.cat([local, var], 1).squeeze(1) for local, var in zip(local_features_set, variance_fingerprint)]
    #
    # local_features = []
    # for set in concatenated_fingerprint:
    #     temp = torch.split(set, 1, dim=2)
    #     temp = [t.squeeze(-1) for t in temp]
    #     local_features.append(temp)

    # print(local_features[0][0].shape)
    # exit()

    return local_features


def load(path):
    _, ext = os.path.splitext(path)
    if ext == '.npy':
        feat = np.load(path)
    elif ext == '.pth':
        feat = torch.load(path)
    else:
        raise TypeError(f'feature extension {ext} isn\'t supported')

    return feat


def load_segment_fingerprint(base_path):
    # base_path
    # ../{dataset}-{decode_rate}-{cnn_extractor}-{group_count}-{aggr_model}/{video}.pth
    # ex) vcdb-5-mobilenet_avg-shot-lstm/00274a.flv.pth

    paths = [os.path.join(base_path, p) for p in os.listdir(base_path)]
    pool = Pool()
    bar = tqdm(range(len(paths)), mininterval=1, ncols=150)
    features = [pool.apply_async(load, args=[p], callback=lambda *a: bar.update()) for p in paths]
    pool.close()
    pool.join()
    bar.close()

    features = [f.get() for f in features]
    length = [f.shape[0] for f in features]
    start = np.cumsum([0] + length)
    # index = np.vstack([start[:-1], start[1:]]).reshape(-1, 2)
    index = np.transpose(np.vstack([start[:-1], start[1:]]))
    return np.concatenate(features), np.array(length), index, paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A. extract local features')

    parser.add_argument('--decode_rate', required=False,  default=1,help="decode rate")
    parser.add_argument('--decode_size', required=False, default=256, help="decode size")
    parser.add_argument('--group_count', required=False,  default=5, help="group count")
    parser.add_argument('--cnn_model', required=False, default='mobilenet', help="cnn model (mobilenet, resnet50)")
    parser.add_argument('--trained', required=False, default=False, help="Whether to use the model trained with triplet loss")
    parser.add_argument('--aggr', required=False, default=False,
                        help="Whether to aggregate frame features")
    parser.add_argument('--feature_path', required=True, default='/nfs_shared_/hkseok/features_local/multiple/vcdb_core-1fps-MobileNet_triplet_sum-5sec',
                        help="feature path")
    parser.add_argument('--video_dataset', required=False,
                        default='/nfs_shared_/hkseok/VCDB/videos/core/',
                        help="video_dataset path")

    args = parser.parse_args()
    print(args)

    decode_rate = args.decode_rate
    decode_size = args.decode_size
    group_count = args.group_count
    cnn_model = None
    aggr_model = None
    pth_dir = args.feature_path
    if not os.path.isdir(pth_dir):
        os.makedirs(pth_dir)

    if args.cnn_model == 'mobilenet':
        cnn_model = MobileNet_local().cuda()
        if args.trained:
            cnn_model.load_state_dict(torch.load('/nfs_shared_/hkseok/mobilenet_avg.pth')['model_state_dict'])
    elif args.cnn_model == 'resnet50':
        cnn_model = Resnet50_local().cuda()
        if args.trained:
            cnn_model.load_state_dict(torch.load('/nfs_shared/MLVD/models/resnet_avg_0_10000_norollback_adam_lr_1e-6_wd_0/saved_model/epoch_3_ckpt.pth')['model_state_dict'])
    cnn_model = nn.DataParallel(cnn_model)

    if args.aggr:
        aggr_model = Local_Maxpooling(group_count)

    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # video = '/nfs_shared/MLVD/VCDB/videos/00274a923e13506819bd273c694d10cfa07ce1ec.flv'
    # local_feature = extract_segment_fingerprint(video,decode_rate,decode_size,transform,cnn_model, group_count, aggr_model)
    # torch.save(local_feature, "testfeatures/test.pth")

    # dst = '/nfs_shared_/hkseok/features_local/multiple-variance'
    # basename = 'vcdb_core-1fps-MobileNet_concat-5sec'
    # pth_dir = os.path.join(dst, basename)  # pth file path
    # if not os.path.isdir(pth_dir):
    #     os.makedirs(pth_dir)

    # video_list = glob.glob('/nfs_shared_/hkseok/VCDB/videos/core/*')
    for idx, video in enumerate(glob.glob(os.path.join(args.video_dataset, '*'))):
        videoname = os.path.basename(video)
        print(videoname, idx)
        local_feature = extract_segment_fingerprint(video, decode_rate, decode_size, transform, cnn_model, group_count, aggr_model)
        dst_path = os.path.join(pth_dir, videoname + '.pth')
        torch.save(local_feature, dst_path)





