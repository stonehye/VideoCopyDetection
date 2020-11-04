import glob
import os
import pickle


# ground = [{'Query':[start, end], 'Ref': [start, end]}, ...]
ground = []

annotation_path = '/nfs_shared/MLVD/VCDB/annotation/'
dst_path = '/nfs_shared_/hkseok/VCDB/'

query_list = glob.glob(annotation_path+'*')
for query in query_list:
    with open(query, "r") as f:
        while True:
            line = f.readline().replace('\n', ',')
            if not line: break
            query_video, ref_video, query_start, query_end, ref_start, ref_end = tuple(line.strip(",").split(","))
            ground.append({query_video: [query_start, query_end], ref_video: [ref_start, ref_end]})

with open(os.path.join(dst_path, 'annotation.pkl'), 'wb') as fw:
    pickle.dump(ground, fw)

print(len(ground))