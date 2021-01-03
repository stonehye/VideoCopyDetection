import subprocess
import os

transform = ['border', 'brightness', 'crop', 'format', 'framerate', 'logo', 'resolution', 'rotate', 'flip', 'grayscale']
level = ['Heavy', 'Medium', 'Light']

# for t in transform:
#     for l in level:
#         video_dataset = os.path.join('/Video_transform/videos2', t, l)
#         feature_path = os.path.join('/nfs_shared_/hkseok/features_local/multiple/simulated_videos3/', t, l)
#         if os.path.isdir(video_dataset):
#             command = '/opt/conda/bin/python -u A_extract_local_features.py --feature_path ' + feature_path + ' --video_dataset ' + video_dataset
#
#             subprocess.call(command, shell=True)

# video_dataset = os.path.join('/Video_transform/videos2', 'origianl_resize')
# feature_path = os.path.join('/nfs_shared_/hkseok/features_local/multiple/simulated_videos3/', 'original')
# if os.path.isdir(video_dataset):
#     command = '/opt/conda/bin/python -u A_extract_local_features.py --feature_path ' + feature_path + ' --video_dataset ' + video_dataset
#
#     subprocess.call(command, shell=True)

for l in range(1, 9):
    l = str(l)
    video_dataset = os.path.join('/Video_transform/tests2', l)
    feature_path = os.path.join('/nfs_shared_/hkseok/features_local/multiple/test_dataset2/', l)
    if os.path.isdir(video_dataset):
        command = '/opt/conda/bin/python -u A_extract_local_features.py --feature_path ' + feature_path + ' --video_dataset ' + video_dataset

        subprocess.call(command, shell=True)