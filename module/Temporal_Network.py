import numpy as np
import sys

sys.setrecursionlimit(10000)



class TN(object):
    def __init__(self, D, I, TEMP_WND=3, MIN_MATCH=3):

        self.TEMP_WND = TEMP_WND
        self.MIN_MATCH = MIN_MATCH

        self.index = I
        self.dist = D

        self.query_length = D.shape[0]

        # isdetect, next_time, next_rank, scores, count
        self.paths = np.zeros((*D.shape, 5), dtype=object)

    def find_linkable_node(self, t, r):
        v_id, f_id = self.index[t, r]
        time, rank = np.where((self.index[t + 1:t + 1 + self.TEMP_WND, :, 0] == v_id) &
                              (f_id < self.index[t + 1:t + 1 + self.TEMP_WND, :, 1]) &
                              (self.index[t + 1:t + 1 + self.TEMP_WND, :, 1] <= f_id + self.TEMP_WND))

        return np.dstack((time + 1 + t, rank)).squeeze(0).tolist()

    def find_max_score_path(self, t, r):
        if self.paths[t, r, 0] != 1:
            nodes = self.find_linkable_node(t, r)
            paths = [[time, rank, *self.find_max_score_path(time, rank)] for time, rank in nodes]
            if len(paths) != 0:
                path = min(paths, key=lambda x: x[-2] / x[-1])
                # path = sorted(paths, key=lambda x: x[-2] / x[-1])[0]
                next_time, next_rank = path[0], path[1]
                score = path[5] + self.dist[t, r]
                count = path[6] + 1
            else:
                next_time, next_rank, score, count = -1, -1, self.dist[t, r], 1
            # print('find', t,r,[1, next_time, next_rank, score])
            self.paths[t, r] = [1, next_time, next_rank, score, count]
        else:
            pass  # print('find-already', t, r, self.paths[t, r])
        return self.paths[t, r]

    def fit(self):
        candidate = []
        for t in range(self.query_length):
            for rank, (v_idx, f_idx) in enumerate(self.index[t]):
                q = [t, t]
                r = [self.index[t, rank, 1], self.index[t, rank, 1]]

                _, next_time, next_rank, score, count = self.find_max_score_path(t, rank)
                while next_time != -1:
                    q[1] = next_time
                    r[1] = self.index[next_time, next_rank, 1]
                    _, next_time, next_rank, _, _ = self.paths[next_time, next_rank]

                if count >= self.MIN_MATCH:
                    candidate.append((v_idx, q,r, score, count))

        candidate = sorted(candidate, key=lambda x: x[-2] / x[-1])
        candidate_video = set()
        nms_candidate = []

        for c in candidate:
            flag = True
            # if c[0] not in candidate_video:
            for nc in nms_candidate:
                if nc[0] == c[0] and (not (nc[1][1] < c[1][0] or c[1][1] < nc[1][0]) and not (
                        nc[2][1] < c[2][0] or c[2][1] < nc[2][0])):
                    flag = False
                    break
            if flag:
                # candidate_video.add(c[0])
                nms_candidate.append(c)

        return nms_candidate


if __name__ == '__main__':
    import faiss

    # load all features
    db_features = np.arange(1, 10).reshape(-1, 1) / 10
    db_features = np.repeat(db_features, 10, axis=1).astype(np.float32)
    print(db_features)

    # table => {db_features_idx : (video id, frame id) ...}
    table = {n: (0, n) if n < 5 else (1, n - 5) for n, v in enumerate(db_features)}
    mapping = np.vectorize(lambda x, table: table[x])

    print(table)

    # extract query video features
    query_video_features = db_features[:5, :]
    print(query_video_features)

    # search top k features per each query frames
    l2index = faiss.IndexFlatL2(db_features.shape[1])
    l2index.add(db_features)
    D, I = l2index.search(query_video_features, k=9)
    print(D)  # dist
    print(I)  # index
    print(mapping(I, table))

    I_to_frame_index = np.dstack(mapping(I, table))  # index to (video id , frame id)

    # find copy segment
    temporal_network = TN(D, I_to_frame_index, 3, 3)
    candidate = temporal_network.fit()

    # [(video_id,[query],[reference],dist,count) ... ]
    print(candidate)