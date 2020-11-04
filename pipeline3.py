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

from module.shotdetect.main import SBD_ffmpeg
from moviepy.editor import VideoFileClip
import cv2
import shutil

import torch
import torchvision


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
            # meta['file_extension'] = track.file_extension
            # meta['format'] = track.format
            # meta['duration'] = float(track.duration) # msec
            # meta['frame_count'] = int(track.frame_count)
            try:
                meta['frame_rate'] = float(track.frame_rate)
            except:
                meta['frame_rate'] = 25.0
        elif track.track_type == 'Video':
            meta['width'] = int(track.width)
            meta['height'] = int(track.height)
            meta['rotation'] = float(track.rotation or 0.)
            meta['codec'] = track.codec
    return meta


def decode_frames_IO(video, meta, size, dst_dir):
    # w, h = (meta['width'], meta['height']) if meta['rotation'] not in [90, 270] else (meta['height'], meta['width'])
    filepath = os.path.join(dst_dir, '%d.jpg')
    command = ['ffmpeg',
               '-hide_banner', '-loglevel', 'panic',
               '-nostdin',
               '-vsync', '2',
               '-i', str(video),
               '-pix_fmt', 'yuvj444p',
               '-vf', 'scale={}:{}'.format(size,size),
               filepath
               ]
    # subprocess.call(command)
    command_string = ' '.join(command)
    os.system(command_string)
    return len(os.listdir(dst_dir))


@torch.no_grad()
def extract_frame_fingerprint(model, loader):

    model.eval()
    frame_fingerprints = []

    for im in loader:
        feature = model(im)
        frame_fingerprints.append(feature.cpu())
    frame_fingerprints = torch.cat(frame_fingerprints)
    return frame_fingerprints


def extract_segment_fingerprint(video, decode_size, transform, cnn_model,aggr_model,group_count, SBD_algorithm):
    # 0. parse video metadata
    meta = parse_metadata(video)

    # 1. decode all frames
    dst_dir = '/nfs_shared_/hkseok/temp' # extracted frame path # TODO
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    meta['frame_count'] = decode_frames_IO(video, meta, decode_size, dst_dir)

    # 2. shot boundary detect
    skip_frame = int(round(meta['frame_rate'] // 2)) # 2fps
    sampled_frames = []
    for idx in range(0, meta['frame_count'], skip_frame):
        frame = cv2.imread(os.path.join(dst_dir, str(idx+1)+'.jpg'))
        sampled_frames.append(frame)
    shot_starts, shot_ends = SBD_ffmpeg(sampled_frames, OPTION=SBD_algorithm)

    if shot_starts == []:
        shot_starts = [0]
        shot_ends = [meta['frame_count']]
        shots = [[0, shot_ends[0]/meta['frame_rate']]]
    else:
        shots = [[(start*skip_frame)/meta['frame_rate'], (end*skip_frame)/meta['frame_rate']] for start, end in zip(shot_starts, shot_ends)]
        shot_starts = [x * skip_frame for x in shot_starts]
        shot_ends = [x * skip_frame for x in shot_ends]
    del sampled_frames

    # 3. Sampling (group_count) frames between shot intervals.
    new_frames = []
    for start, end in zip(shot_starts, shot_ends):
        window = list(range(start, end+1))
        count = len(window)
        if group_count == 1:
            divide_interval = 1
        else:
            divide_interval = (count - 2) / (group_count - 1)
        if divide_interval < 1:
            new_frames += window
            remainder = group_count - len(window)
            new_frames += [window[-1]] * remainder
        else:
            divide_interval = int(divide_interval)
            new_frames.append(window[0])
            temp_cnt = group_count - 1
            for idx, tp in enumerate(window[1:]):
                if temp_cnt == 0:
                    break
                if len(window[idx:]) < divide_interval or (idx + 1) % divide_interval == 0:
                    new_frames.append(tp)
                    temp_cnt -= 1

    narray_frames = []
    for idx in new_frames:
        frame = cv2.imread(os.path.join(dst_dir, str(idx + 1) + '.jpg'))
        if frame is not None:
            narray_frames.append(frame)

    # convert from PIL to narray
    frames = []
    for frame in narray_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(frame.astype('uint8'), 'RGB')
        frames.append(PIL_image)
    del narray_frames

    # 4. extract frame fingerprint
    cnn_loader = DataLoader(ListDataset(frames, transform=transform), batch_size=64, shuffle=False, num_workers=4)
    frame_fingerprints = extract_frame_fingerprint(cnn_model, cnn_loader)
    frame_fingerprints = frame_fingerprints.reshape(-1,frame_fingerprints.shape[1],frame_fingerprints.shape[2]*frame_fingerprints.shape[2],1)
    print(frame_fingerprints.size())

    # # grouping fingerprints for each segment => If frame_fingerprints cannot be divided by group_count, the last is copied.
    # k = group_count - frame_fingerprints.shape[0] % group_count
    # if k != group_count:
    #     frame_fingerprints = torch.cat([frame_fingerprints, frame_fingerprints[-1:, ].repeat((k, 1))])
    # frame_fingerprints = frame_fingerprints.reshape(-1, group_count, frame_fingerprints.shape[2])
    #
    # # 5. extract segment_fingerprint
    # frame_fingerprints = frame_fingerprints.permute(0, 2, 1)
    # segment_fingerprints = aggr_model(frame_fingerprints)
    #
    # del frame_fingerprints
    # shutil.rmtree(dst_dir)
    #
    # return segment_fingerprints, shots


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
    """
        Make segment features of the entire video into one feature file
    """
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
    index = np.transpose(np.vstack([start[:-1], start[1:]]))
    return np.concatenate(features), np.array(length), index, paths


if __name__ == '__main__':
    video = '/nfs_shared/MLVD/VCDB/videos/5df28e18b3d8fbdc0f4cd07ef5aefcdc1b4f8d42.flv'
    decode_size = 256
    group_count = 10
    cnn_model = MobileNet_local().cuda()
    # cnn_model.load_state_dict(torch.load('/nfs_shared_/hkseok/mobilenet_avg.pth')['model_state_dict'])
    cnn_model = nn.DataParallel(cnn_model)
    aggr_model = Segment_Maxpooling()
    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    extract_segment_fingerprint(video, decode_size, transform, cnn_model, aggr_model, group_count, 'local')