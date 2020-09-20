import os
import logging
import json
from datetime import datetime, timedelta, timezone
from utils.period import Period, CustomEncoder

from math import ceil

kst = timezone(timedelta(hours=9))


def get_dict_without_key(d, keys):
    if isinstance(keys, str):
        keys = [keys]
    return {k: v for k, v in d.items() if k not in keys}


def get_dict_with_key(d, keys):
    if isinstance(keys, str):
        keys = [keys]
    return {k: v for k, v in d.items() if k in keys}


def init_logger(name, log_dir='/workspace/log'):
    logger = logging.getLogger(__name__)
    log_dir = os.path.join(log_dir, now_ymd())
    name = os.path.join(log_dir, *[i for i in name.split(os.sep) if i not in ('..', '.')])

    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(name)
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)
    logger.setLevel(level=logging.INFO)

    logger.info("=========================================================")
    logger.info("[{time}] Log start ... {path}".format(time=datetime.now(tz=kst).strftime("%Y-%m-%d %H:%M:%S"),
                                                       path=os.path.abspath(name)))
    logger.info("=========================================================")
    return logger, name


def end_logger(logger, msg=''):
    logger.info("=========================================================")
    logger.info("[{time}] END - {msg}".format(time=datetime.now(tz=kst).strftime("%Y-%m-%d %H:%M:%S"), msg=msg))
    logger.info("=========================================================")


def now_ymd():
    return datetime.now(tz=kst).strftime('%-y%m%d')


def sec_format(time):
    if isinstance(time, int):
        return timedelta(seconds=time)
    elif isinstance(time, float):
        return timedelta(seconds=ceil(time))
    elif isinstance(time, timedelta):
        return timedelta(seconds=ceil(time.total_seconds()))


def json_dump(json_file, data):
    prev = []
    if os.path.exists(json_file) and os.stat(json_file).st_size != 0:
        with open(json_file, 'r') as fp:
            prev = json.load(fp)
    with open(json_file, 'w') as fp:
        data.extend(prev)
        json.dump(data, fp, cls=CustomEncoder)


if __name__ == '__main__':
    print(os.path.relpath('asdasd/out.txt', '/workspace/log'))