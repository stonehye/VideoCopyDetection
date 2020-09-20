from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms as trn
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import warnings
import imageio
import os
import shutil
import random
import cv2
from PIL import Image
import subprocess
import imageio_ffmpeg
import imageio
import json
from typing import Union
import copy
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors.content_detector import ContentDetector

'''
VCDB root dir 
root
|-- annotation
|   |-- baggio_penalty_1994.txt
|   `-- ...
|-- background_dataset
|   |-- frames -> /DB/VCDB/background_dataset/frames/
|   `-- videos -> /DB/VCDB/background_dataset/videos/
`-- core_dataset    
    |-- frames -> /DB/VCDB/frames
    `-- videos -> /DB/VCDB/core_dataset   
'''


class VCDB(Dataset):
    def __init__(self, root='/DB/VCDB'):
        self.root = root
        self.core_dir = os.path.join(self.root, 'core_dataset')
        self.background_dir = os.path.join(self.root, 'background_dataset')
        self.annotation_dir = os.path.join(self.root, 'annotation')

        self.nfold = 1
        self.valid_idx = 0
        self.segment_length = 1
        self.frames_per_segment = 1

        self.core, self.core_table = self._scan_videos(self.core_dir, 'videos', 'frames', 'meta', '.meta')

        self.core_train = []
        self.core_valid = copy.deepcopy(self.core)
        self.background,self.bg_table = self._scan_videos(self.background_dir, 'videos', 'frames', 'meta', '.meta',max_cnt=-1)

        self.annotation, self.segment, self.core_labels = self._scan_annotation()

        self.label_valid = set(self.core_labels)
        self.label_train = set()
        self.annotation_train = []
        self.annotation_valid = copy.deepcopy(self.annotation)
        self.segment_train = []
        self.segment_valid = copy.deepcopy(self.segment)

    def _scan_annotation(self):
        seg_key = ['label', 'name', 'start', 'end']
        annotation = []
        labels = set()
        segment = set()
        for n in sorted(os.listdir(self.annotation_dir)):
            label, _ = os.path.splitext(n)
            with open(os.path.join(self.annotation_dir, n), 'r') as f:
                lines = list(map(lambda x: x.strip().split(','), f.readlines()))
                for line in lines:
                    line[2:] = list(map(self._time2sec, line[2:]))
                    seg1 = (label, line[0], *line[2:4])
                    seg2 = (label, line[1], *line[4:])
                    segment.add(seg1)
                    segment.add(seg2)
                    annotation.append(({k: v for k, v in zip(seg_key, seg1)},
                                       {k: v for k, v in zip(seg_key, seg2)}))
                    labels.add(label)

        segment = [dict(zip(seg_key, q)) for q in segment]
        segment.sort(key=lambda x: x['end'], reverse=True)
        segment.sort(key=lambda x: (x['label'], x['name'], x['start']))
        return annotation, segment, labels

    def _time2sec(self, time):
        return sum([int(i) * 60 ** (2 - n) for n, i in enumerate(time.split(':'))])

    def _extract_frame(self, video_path, frame_dir):
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        assert len(os.listdir(frame_dir)) == 0

        cmd = 'ffmpeg -i {} -map 0:v:0 -q:v 0 -vsync 2 -f image2 {}/%6d.jpg'.format(video_path, frame_dir).split(' ')
        p = subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        out, err = p.communicate()
        if p.returncode == 1:
            raise ChildProcessError('Decode failed .... {}'.format(video_path))
        return p.returncode, out.decode('utf8'), err.decode('utf8'), cmd

    def _parse_video(self, v_path, f_path):
        assert os.path.exists(v_path)
        if not os.path.exists(f_path) or not len(os.listdir(f_path)):
            self._extract_frame(v_path, f_path)
        nframes = len(os.listdir(f_path))
        duration = imageio.get_reader(v_path, 'ffmpeg', loop=True).get_meta_data()['duration']
        # scene_list = self._scene_detect(v_path, downscale=True)
        data = {'fps': round(nframes / duration, 2),
                'duration': duration,
                'nframes': nframes}
        # 'scene': scene_list,
        # 'scene_count': len(scene_list)}
        return data

    def _create_metadata(self, meta_path, video_root, frame_root, max_cnt=None):
        metadata = []
        c = 1
        for label in sorted(os.listdir(video_root)):
            for v in sorted(os.listdir(os.path.join(video_root, label))):
                v_path = os.path.join(video_root, label, v)
                f_path = os.path.join(frame_root, label, os.path.splitext(v)[0])
                data = self._parse_video(v_path, f_path)
                metadata.append({'name': v, 'label': label, **data})
                if c == max_cnt: break
                c += 1
        with open(meta_path, 'w') as m:
            json.dump(metadata, m)

    def _read_metadata(self, meta_path):
        with open(meta_path, 'r') as m:
            metadata = json.load(m)
        return metadata

    def _scan_videos(self, base_dir, video='videos', frame='frames', meta='meta', meta_ext='.meta', max_cnt=-1):
        videos = []
        table = dict()
        video_root = os.path.join(base_dir, video)
        frame_root = os.path.join(base_dir, frame)
        meta_root = os.path.join(base_dir, meta)
        c = 1
        for label in sorted(os.listdir(video_root)):
            for v in sorted(os.listdir(os.path.join(video_root, label))):
                v_path = os.path.join(video_root, label, v)
                f_path = os.path.join(frame_root, label, os.path.splitext(v)[0])
                m_path = os.path.join(meta_root, label, v + meta_ext)
                if not os.path.exists(m_path):
                    if not os.path.exists(os.path.dirname(m_path)):
                        os.makedirs(os.path.dirname(m_path))
                    print('parse video meta data .. {} - {}'.format(c, m_path))
                    data = {'name': v, 'label': label, **self._parse_video(v_path, f_path)}
                    with open(m_path, 'w') as m:
                        json.dump(data, m)
                else:
                    data = self._read_metadata(m_path)
                videos.append(data)
                table['{label}/{name}'.format(label=data['label'], name=data['name'])] = data
                if c == max_cnt:
                    return videos, table
                c += 1


        return videos, table

    def _scene_detect(self, video, downscale=False):
        print(video)
        video_manager = VideoManager([video])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(ContentDetector(threshold=20))

        base_timecode = video_manager.get_base_timecode()

        if downscale:
            video_manager.set_downscale_factor()

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = [(scene[0].get_frames(), scene[1].get_frames()) for scene in
                      scene_manager.get_scene_list(base_timecode)]
        video_manager.release()
        print(scene_list)
        return scene_list

    def get_annotation(self, **kwargs):
        l = self.annotation
        if "label" in kwargs:
            label = kwargs.get('label')
            if isinstance(label, str):
                label = [label]
            l = [ann for ann in l if ann[0]['label'] in label or ann[1]['label'] in label]

        if "name" in kwargs:
            name = kwargs.get('name')
            if isinstance(name, str):
                l = [ann for ann in l if ann[0]['name'] == name or ann[1]['name'] == name]
            elif isinstance(name, list) and len(name) == 1:
                l = [ann for ann in l if ann[0]['name'] == name[0] or ann[1]['name'] == name[0]]
            elif isinstance(name, list) and len(name) == 2:
                ret = []
                for ann in l:
                    if ann[0]['name'] == name[0] and ann[1]['name'] == name[1]:
                        ret.append(ann)
                    elif ann[1]['name'] == name[0] and ann[0]['name'] == name[1]:
                        ret.append(ann[::-1])
                l = ret

        if 'video' in kwargs and kwargs.get('video'):
            l = [({**ann[0], **self.core_table['{label}/{name}'.format(label=ann[0]['label'], name=ann[0]['name'])]},
                  {**ann[1], **self.core_table['{label}/{name}'.format(label=ann[1]['label'], name=ann[1]['name'])]})
                 for ann in l]

        return copy.deepcopy(l)

    def get_segment(self, **kwargs):
        l = self.segment
        if "label" in kwargs:
            label = kwargs.get('label')
            if isinstance(label, str):
                label = [label]
            l = [seg for seg in l if seg['label'] in label]
        if "name" in kwargs:
            name = kwargs.get('name')
            if isinstance(name, str):
                name = [name]
            l = [seg for seg in l if seg['name'] in name]

        return l

    def get_core_videos(self, **kwargs):
        if 'valid' in kwargs and kwargs['valid'] == False:
            l = copy.deepcopy(self.core_train)
        elif 'train' in kwargs and kwargs['train'] == False:
            l = copy.deepcopy(self.core_valid)
        else:
            l = copy.deepcopy(self.core)

        if "label" in kwargs:
            label = kwargs.get('label')
            if isinstance(label, str):
                label = [label]
            l = [vid for vid in l if vid['label'] in label]
        if "name" in kwargs:
            name = kwargs.get('name')
            if isinstance(name, str):
                name = [name]
            l = [vid for vid in l if vid['name'] in name]

        return l

    def get_background_videos(self, cnt):
        assert len(self.background) >= cnt
        bi = np.random.choice(len(self.background), size=cnt, replace=False)
        return [self.background[i] for i in bi]

    def get_reference_videos(self, background=False, total=528):
        # assert valid or train or background
        l = self.get_core_videos(train=False)
        core_cnt = len(l)
        if background and core_cnt < total:
            bg = self.get_background_videos(total - core_cnt)
            l += bg
        return l, core_cnt

    def set_vcdb_parameters(self, nfold=1, valid_idx=0, segment_length=1, frames_per_segment=1):
        self.set_fold(nfold=nfold, valid_idx=valid_idx)
        self.set_segment(segment_length=segment_length, frames_per_segment=frames_per_segment)

    def set_fold(self, nfold=1, valid_idx=0):
        assert 0 <= valid_idx < nfold
        self.nfold = nfold
        self.valid_idx = valid_idx

        seg_count = {k: 0 for k in self.core_labels}
        for seg in self.segment:
            seg_count[seg['label']] += 1

        seg_count = [k for k, v in sorted(seg_count.items(), key=lambda x: x[1], reverse=True)]

        self.label_valid = sorted({seg_count[i] for i in range(valid_idx, len(seg_count), nfold)})
        self.label_train = sorted({g for g in seg_count if g not in self.label_valid})

        self.core_train = self.get_core_videos(label=self.label_train)
        self.core_valid = self.get_core_videos(label=self.label_valid)

        self.annotation_train = self.get_annotation(label=self.label_train)
        self.annotation_valid = self.get_annotation(label=self.label_valid)

        self.segment_train = self.get_segment(label=self.label_train)
        self.segment_valid = self.get_segment(label=self.label_valid)

    def set_segment(self, segment_length: Union[int, str] = 1, frames_per_segment: int = 1):
        self.segment_length = segment_length
        self.frames_per_segment = frames_per_segment

        # if segment_length=='scene' : scan .scene files

    def get_segment_frames(self, video):
        root = self.core_dir if video['label'] in self.core_labels else self.background_dir
        f_path = os.path.join(root, 'frames', video['label'], os.path.splitext(video['name'])[0])
        frames = sorted(os.listdir(f_path))
        idx = []
        if isinstance(self.segment_length, str) and self.segment_length.lower() == 'scene':
            seg = video['scene']
            for start, end in seg:
                intv = (end - start) / self.frames_per_segment
                # idx.append([int(round(i)) for i in np.arange(start + intv / 2, end - 1, intv)])
                idx.append([i for i in np.arange(start + intv / 2, end - 1, intv)])

        elif isinstance(self.segment_length, int):
            intv = video['fps'] * self.segment_length / self.frames_per_segment
            idx = [int(round(i)) for i in np.arange(intv / 2, video['nframes'] - 1, intv)]
            idx = [idx[i:i + self.frames_per_segment] for i in range(0, len(idx), self.frames_per_segment)]
            if len(idx[-1]) != self.frames_per_segment:
                [idx[-1].append(idx[-1][-1]) for i in range(0, self.frames_per_segment - len(idx[-1]))]

        return [[os.path.join(f_path, frames[i]) for i in seg] for seg in idx]

    def toDict(self):
        return {'name': self.__class__.__name__,
                'root': self.root,
                'annotation': {'pair': len(self.annotation), 'segments': len(self.segment)},
                'core': {'dir': self.core_dir, 'labels': len(self.core_labels), 'videos': len(self.core)},
                'background': {'dir': self.background_dir, 'videos': len(self.background)},
                'Nfold': self.nfold,
                'valid_index': self.valid_idx,
                'segment_length': self.segment_length,
                'frames_per_segment': self.frames_per_segment,
                'Train': {'labels_count': len(self.label_train), 'labels': list(self.label_train),
                          'videos': len(self.core_train),
                          'annotation': len(self.annotation_train), 'segment': len(self.segment_train)},
                'Valid': {'labels_count': len(self.label_valid), 'labels': list(self.label_valid),
                          'videos': len(self.core_valid),
                          'annotation': len(self.annotation_valid), 'segment': len(self.segment_valid)}

                }

    def toStr(self):
        out = 'VCDB Dataset\n'
        out += '>> root ... {}\n'.format(self.root)
        out += '>> Annotation ... {} pairs, {} segments\n'.format(len(self.annotation), len(self.segment))
        out += '>> core ... {} videos, {} labels\n'.format(len(self.core), len(self.core_labels))
        out += '>> background ... {} videos\n'.format(len(self.background))
        out += '>> N Fold ... {}\n'.format(self.nfold)
        out += '>> Valid Idx ... {}\n'.format(self.valid_idx)
        out += '>> Segment Length ... {}\n'.format(self.segment_length)
        out += '>> Frames Per Segment ... {}\n'.format(self.frames_per_segment)
        out += '>> Train ... {} labels, {} videos, {} pairs, {} segments\n'.format(len(self.label_train),
                                                                                   len(self.core_train),
                                                                                   len(self.annotation_train),
                                                                                   len(self.segment_train))
        out += '>> Train label ... {}\n'.format(list(self.label_train))

        out += '>> Valid ... {} labels, {} videos, {} pairs, {} segments\n'.format(len(self.label_valid),
                                                                                   len(self.core_valid),
                                                                                   len(self.annotation_valid),
                                                                                   len(self.segment_valid))
        out += '>> Valid Group ... {}'.format(list(self.label_valid))

        return out


class NestedListDataset(Dataset):
    # stack nest list items
    # l = [[a.jpg,b.jpg,c.jpg],[d.jpg,e.jpg,f.jpg] ... ]
    def __init__(self, l):
        self.l = l
        self.loader = default_loader  # self.feature_loader
        self.default_trn = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # return (n, images) shape: [n,3,224,224]
    def __getitem__(self, idx):
        path = self.l[idx]
        frames = [self.default_trn(self.loader(p)) for p in path]
        frame = torch.stack(frames)
        path = ','.join(path)

        return path, frame

    def __len__(self):
        return len(self.l)


if __name__ == '__main__':
    import pprint
    db = VCDB()
    db.set_vcdb_parameters(4,0,5,1)

    db.get_reference_videos()



    exit()
    a = db.get_annotation(label={'the_pursuit_of_happyness_-_job_interview', 'brazil_vs_brazil_nike_commercial_2012',
                                 'david_beckham_lights_the_olympic_torch'})
    print(a)

    # db.set_segment(segment_length=5, frames_per_segment=2)
    # vid = db.get_core_videos(name='750357acf34899c1c99eb2ae22880c88a776c946.flv')
    # print(vid)

    # print(db.get_core_videos())

    a = db.get_annotation(
        name=['46f2e964ae16f5c27fad70d6849c76616fad7502.flv', '46f2e964ae16f5c27fad70d6849c76616fad7502.flv'])
    print(*a, sep='\n')