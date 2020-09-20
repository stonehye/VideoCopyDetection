import json
import numpy as np
from collections import namedtuple

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if "tojson" in dir(o):
            return o.tojson()
        return json.JSONEncoder.default(self, o)

class CustomDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, o):
        if "fromjson" in dir(o):
            return o.fromjson()




class Period(object):
    # closed-form [ )
    def __init__(self, start=-1, end=-1):
        if isinstance(start,np.int64):
            start=int(start)
        if isinstance(end,np.int64):
            end=int(end)

        (self.start, self.end) = (start, end) if start < end else (end, start)

    def __repr__(self):
        return '{} - {}'.format(self.start, self.end)

    def __len__(self):
        return self.end - self.start

    def tojson(self):

        return {'start':self.start,'end':self.end}

    def fromjson(self,o):
        print(fr)
        return namedtuple('X', o.keys())(*o.values())



    def is_overlap(self, p):
        if not isinstance(p, Period):
            raise ValueError('p is not Period object')

        return not (self.end < p.start or p.end < self.start)

    def merge_with(self, p):
        self.start = self.start if self.start < p.start else p.start
        self.end = self.end if self.end > p.end else p.end

    def is_contain(self, p):
        return self.start <= p.start and p.end <= self.end

    def intersection(self, p):
        if not isinstance(p, Period):
            raise ValueError('p is not Period object')

        if not self.is_overlap(p): return False

        max_start = max(self.start, p.start)
        min_end = min(self.end, p.end)
        return Period(max_start, min_end)

    def union(self, p):
        if not isinstance(p, Period):
            raise ValueError('p is not Period object')

        if not self.is_overlap(p): return False

        min_start = min(self.start, p.start)
        max_end = max(self.end, p.end)

        return Period(min_start, max_end)

    def IOU(self, p):
        try:
            intersect = self.intersection(p)
            union = self.union(p)
            iou = len(intersect) / len(union)
        except:
            iou = 0

        return iou

    def isin(self, p):
        if not isinstance(p, Period):
            raise ValueError('p is not Period object')
        return (p.start <= self.start) and (self.end <= p.end)

    def is_replace(self, p):
        if not isinstance(p, Period):
            raise ValueError('p is not Period object')
        return (self.start <= p.start) and (p.end <= self.end)

    def __sub__(self, p):
        if not isinstance(p, Period):
            raise ValueError('p is not Period object')

        if self.is_overlap(p):
            max_start = max(self.start, p.start)
            min_end = min(self.end, p.end)
            intersect = Period(max_start, min_end)


if __name__ == '__main__':
    p = Period(137.0, 141)
    b = Period(33, 0.4)

    print(json.dumps(b, indent=4, cls=CustomEncoder))
    exit()

    print(p, b, p.is_overlap(b))
    print(p.__dict__)

    exit()
    l = [[1], [2], [3], [4], [5]]
    for n, i in enumerate(l):
        if i == [2]: l[2] = -1
    print(l)
    exit()
    a = Period(0, 8)
    b = Period(0, 52)
    print(a.is_overlap(b))

    print(a.intersection(b))
    print(a.union(b))
    print(a.IOU(b))

    print(Period.IOU(a, b))