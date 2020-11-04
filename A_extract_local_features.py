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
    print(frame_fingerprints.shape)

    if group_count != 1:
        ## multiple keyframe
        # grouping fingerprints for each segment => If frame_fingerprints cannot be divided by group_count, the last is copied.
        k = group_count - frame_fingerprints.shape[0] % group_count
        if k != group_count:
            frame_fingerprints = torch.cat([frame_fingerprints, frame_fingerprints[-1:, ].repeat((k, 1, 1))])
        frame_fingerprints = frame_fingerprints.reshape(-1, group_count * frame_fingerprints.shape[1], frame_fingerprints.shape[-1])
        print(frame_fingerprints.shape)

    if aggr_model:
        ## multiple keyframe - segment local maxpooling
        frame_fingerprints = aggr_model(frame_fingerprints)
        print(frame_fingerprints.shape)

    local_features = []
    local_features_set = torch.split(frame_fingerprints, 1)
    for set in local_features_set:
        temp = torch.split(set, 1, dim=2)
        temp = [t.squeeze(-1) for t in temp]
        local_features.append(temp)

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
    video = '/nfs_shared/MLVD/VCDB/videos/00274a923e13506819bd273c694d10cfa07ce1ec.flv'
    decode_rate = 0.2
    decode_size = 400
    group_count = 1
    cnn_model = MobileNet_local_middle().cuda()
    cnn_model = nn.DataParallel(cnn_model)
    # aggr_model = Local_Maxpooling()
    aggr_model = None
    transform = trn.Compose([
        trn.Resize((400, 400)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # local_feature = extract_segment_fingerprint(video,decode_rate,decode_size,transform,cnn_model, group_count)
    #
    # torch.save(local_feature, "testfeatures/test.pth")

    dst = '/nfs_shared_/hkseok/features_local/single'
    basename = 'vcdb_core_train-1fps-mobilenet_local_middle-5sec-400size'
    pth_dir = os.path.join(dst, basename)  # pth file path
    if not os.path.isdir(pth_dir):
        os.makedirs(pth_dir)
    video_list = glob.glob('/nfs_shared_/hkseok/VCDB/train/*')
    for idx, video in enumerate(video_list):
        videoname = os.path.basename(video)
        print(videoname, idx)
        local_feature = extract_segment_fingerprint(video, decode_rate, decode_size, transform, cnn_model, group_count, aggr_model)
        dst_path = os.path.join(pth_dir, videoname + '.pth')
        torch.save(local_feature, dst_path)





