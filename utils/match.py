'''
pascal VOC - detect : ground
1:N -> IOU가 가장높은 1개만 true, 나머지는 false
https://github.com/rafaelpadilla/Object-Detection-Metrics/issues/2

N:1 -> Multiple detections of the same object in an image are considered false detections
e.g. 5 detections of a single object is counted as 1 correct detection and 4 false detections
http://host.robots.ox.ac.uk/pascal/VOC/voc2010/devkit_doc_08-May-2010.pdf
https://github.com/rafaelpadilla/Object-Detection-Metrics

1. IOU 기준으로 TP / FP check
2. class 별로 Confidence 를 기준으로 sort
3. AP 구하기
=> https://github.com/rafaelpadilla/Object-Detection-Metrics/issues/78
'''


def match(detect, ground):
    # detect:ground 가 1:N 인 경우 -> N true(문제 x)
    # detect:ground 가 N:1 인 경우 -> X , detect는 1개일때만 가정
    result = {'detect': {'hit': [], 'miss': []}, 'ground': {'hit': [], 'miss': []}}

    for d in detect:
        isHit = False
        for g in ground:
            if d['Ref'].is_overlap(g['Ref']) and d['Query'].is_overlap(g['Query']):
                result['detect']['hit'].append(d)
                result['ground']['hit'].append(g)
                isHit = True
        if not isHit:
            result['detect']['miss'].append(d)

    result['ground']['miss'] = [g for g in ground if g not in result['ground']['hit']]

    return result


if __name__ == '__main__':
    a = {'a': 1, 'b': 2}
    b = a.copy()
    # b['a'].append(2)
    b['a'] = 3
    print(id(a), a)
    print(id(b), b)