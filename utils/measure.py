from datetime import timedelta
from utils.utils import sec_format


class Measure(object):
    '''
    result = {'Query': q_vid,
                  'TP_count': 0, 'TP': [], 'TP_GT': [],
                  'FP_count': 0, 'FP': [],
                  'FN_count': 0, 'FN': [],
                  }
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.feature_extract_time = 0
        self.ref_fp_cnt = 0

        self.tp, self.fp, self.fn = 0, 0, 0
        self.query_time = 0
        self.results = dict()
        self.q_cnt=0

    def update(self, r):
        qvid = r['Query']
        key = '{}/{}'.format(qvid['label'], qvid['name'])

        if self.results.get(key) is None:
            self.results[key] = {'Query': qvid,
                                 'TP_count': 0, 'TP': [], 'TP_GT': [],
                                 'FP_count': 0, 'FP': [],
                                 'FN_count': 0, 'FN': [],
                                 'query_time': 0,
                                 }
            self.q_cnt+=1

        self.results[key]['TP_count'] += len(r['detect']['hit'])
        self.results[key]['TP'].extend(r['detect']['hit'])
        self.results[key]['TP_GT'].extend(r['ground']['hit'])
        self.results[key]['FP_count'] += len(r['detect']['miss'])
        self.results[key]['FP'].extend(r['detect']['miss'])
        self.results[key]['FN_count'] += len(r['ground']['miss'])
        self.results[key]['FN'].extend(r['ground']['miss'])
        self.results[key]['query_time'] += r['query_time']

        self.tp += len(r['detect']['hit'])
        self.fp += len(r['detect']['miss'])
        self.fn += len(r['ground']['miss'])
        self.query_time += r['query_time']

    def _eval_performance(self, tp, fp, fn, eps=1e-6):
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = (2 * prec * rec) / (prec + rec + eps)
        return f1, prec, rec

    def eval_performance(self):
        return self._eval_performance(self.tp, self.fp, self.fn), (self.tp, self.fp, self.fn), self.query_time

    def eval_query_performance(self, qvid):
        key = '{}/{}'.format(qvid['label'], qvid['name'])
        tp = self.results[key]['TP_count']
        fp = self.results[key]['FP_count']
        fn = self.results[key]['FN_count']
        time = self.results[key]['query_time']
        return self._eval_performance(tp, fp, fn), (tp, fp, fn), time

    def get_result(self, qvid):
        key = '{}/{}'.format(qvid['label'], qvid['name'])
        return self.results[key]

    def get_result_detail_str(self, qvid, indent=2):
        result = self.get_result(qvid)
        tab = ' ' * indent
        out_str = '{tab}TP : {}\n'.format(result['TP_count'], tab=tab * 2)
        out_str += ''.join(['{tab}{}\n'.format(d, tab=tab * 3) for d in result['TP']])
        out_str += '{tab}TP_GT : {}\n'.format(result['TP_count'], tab=tab * 2)
        out_str += ''.join(['{tab}{}\n'.format(d, tab=tab * 3) for d in result['TP_GT']])
        out_str += '{tab}FP : {}\n'.format(result['FP_count'], tab=tab * 2)
        out_str += ''.join(['{tab}{}\n'.format(d, tab=tab * 3) for d in result['FP']])
        out_str += '{tab}FN : {}\n'.format(result['FN_count'], tab=tab * 2)
        out_str += ''.join(['{tab}{}\n'.format(d, tab=tab * 3) for d in result['FN']])

        return out_str.rstrip()

    def get_query_performance_str(self, qvid, indent=2):
        tab = ' ' * indent
        (f1, prec, rec), (tp, fp, fn), time = self.eval_query_performance(qvid)
        (tf1, tprec, trec), (ttp, tfp, tfn), ttime = self.eval_performance()

        out = ['F-score: {:.4f}({:.4f})'.format(f1, tf1),
               'Precision: {:.4f}({:.4f})'.format(prec, tprec),
               'Recall: {:.4f}({:.4f})'.format(rec, trec),
               'TP: {}({})'.format(tp, ttp),
               'FP: {}({})'.format(fp, tfp),
               'FN: {}({})'.format(fn, tfn),
               'query_time: {}({})'.format(sec_format(time),
                                           sec_format(ttime / self.q_cnt))]

        return '{tab}{}'.format(', '.join(out), tab=tab)

    def get_query_str(self, qvid, indent=2):
        return '{}\n{}'.format(self.get_query_performance_str(qvid, indent),
                               self.get_result_detail_str(qvid, indent))

    def get_performance_dict(self):
        (f1, prec, rec),(tp,fp,fn),q_time = self.eval_performance()
        return {'F-score': f1, 'Precision': prec, 'Recall': rec, 'TP': tp, 'FP': fp, 'FN': fn,
                'Query_time': q_time, 'Extract_time': self.feature_extract_time}

    def get_performance_str(self):
        performance = self.get_performance_dict()

        out = ['F-score: {:.4f}'.format(performance['F-score']),
               'Precision: {:.4f}'.format(performance['Precision']),
               'Recall:{:.4f}'.format(performance['Recall']),
               'TP: {}'.format(performance['TP']),
               'FP: {}'.format(performance['FP']),
               'FN: {}'.format(performance['FN']),
               'query_time: {}({})'.format(sec_format(performance['Query_time']),
                                           sec_format(performance['Query_time'] / self.q_cnt)),
               'extract_time: {}'.format(sec_format(performance['Extract_time']))]
        return ', '.join(out)
