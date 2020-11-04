import pickle as pk
import numpy as np
from collections import defaultdict
import csv
import pandas as pd

name = 'fourth2'

result = pk.load(open(f'{name}.pkl', 'rb'))

# print(result)


out = []

recall_csv = f'{name}-recall.csv'
recall = []
histogram = f'{name}-histogram.csv'

result_per_feature = dict()
for qv, ret in result.items():
    result_per_feature[qv] = defaultdict(list)
    for rf, ranks in ret.items():
        for r in ranks:
            for i in r:
                result_per_feature[qv][i[0]].append(i[2])
                out.append(i[2])
# print(result_per_feature)
out = np.array(out)
print(out)
total = out.shape[0]

max = np.max(out)
print(max)

a = 1
r = np.where(out < a)[0].shape[0]
# print(f'recall at {a} : {r/total:.4f} {r}/{total}')
recall.append({'topk': a, 'recall': r / total, 'count': r})
print(r)
for a in range(10, 1000, 10):
    r = np.where(out < a)[0].shape[0]
    # print(f'recall at {a} : {r/total:.4f} {r}/{total}')
    recall.append({'topk': a, 'recall': r / total, 'count': r})

for a in range(1000, 10000, 1000):
    r = np.where(out < a)[0].shape[0]
    # print(f'recall at {a} : {r/total:.4f} {r}/{total}')
    recall.append({'topk': a, 'recall': r / total, 'count': r})

for a in range(10000, 110000, 10000):
    r = np.where(out < a)[0].shape[0]
    # print(f'recall at {a} : {r/total:.4f} {r}/{total}')
    recall.append({'topk': a, 'recall': r / total, 'count': r})

recall = pd.DataFrame(recall)
recall.to_csv(recall_csv, index=False)

hist=pd.DataFrame(np.bincount(out))
print(hist)
hist.to_csv(histogram)


