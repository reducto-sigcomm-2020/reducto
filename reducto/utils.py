import sys
import time

import numpy as np
import pandas as pd


def user_prompt(question='Are you sure to continue?'):
    """ Prompt the yes/no-*question* to the user. """
    from distutils.util import strtobool
    while True:
        user_input = input(question + ' [Y/n]: ').lower()
        try:
            result = strtobool(user_input)
            return result
        except ValueError:
            print('Please use y/n or yes/no\n')


def human_readable_size(size, decimal_places=3):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f'{size:.{decimal_places}f}{unit}'


def timeit(method, header, filename):
    def timed_func(*args, **kwargs):
        time_start = time.time()
        result = method(*args, **kwargs)
        time_end = time.time()
        time_used = time_end - time_start
        print(f'{filename},{header},'
              f'time_start={time_start},time_end={time_end},time_used={time_used}')
        return result
    return timed_func


def timeit2(method):
    def timed_func(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        ms = (te - ts) * 1000
        if 'log_time' in kw:
            kw['log_time'][kw.get('log_name', method.__name__.upper())] = ms
        else:
            print('{} {:2.2f} ms'.format(method.__name__, (te - ts) * 1000))
        return result
    return timed_func


def redirect(stdout=sys.stdout, stderr=sys.stderr):
    def wrap(f):
        def newf(*args, **kwargs):
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = stdout
            sys.stderr = stderr
            try:
                return f(*args, **kwargs)
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
        return newf
    return wrap


def flatten(lists: [[]]) -> []:
    return [item for sublist in lists for item in sublist]


def assert_list(items, typ):
    if isinstance(items, typ):
        return [items]
    assert isinstance(items, list)
    if len(items) == 0:
        return []
    assert all(isinstance(item, typ) for item in items)
    return items


def generate_thresholds(diff_vectors, num_thresholds=50):
    thresholds = {}

    dv_record = {}
    for seg_info in diff_vectors:
        for dp, dv in seg_info.items():
            if dp not in dv_record:
                dv_record[dp] = []
            dv_record[dp] += dv

    # Get histogram
    for dp, record in dv_record.items():
        # Use density of histogram to determine # thersholds in each interval
        record = np.array(record)
        equal_space = np.linspace(
            record.min(),
            record.max(),
            int(num_thresholds/4),
            endpoint=True
        )
        hist = np.histogram(record, equal_space)[0]
        hist = hist / len(record)
        thresh_n = np.ceil(hist * num_thresholds)

        # Force the number of thresholds is equal to the arg num_thresholds
        total_thresh = sum(thresh_n)
        index = 0
        while total_thresh > num_thresholds:
            # NOTE Remove the thresh from smallest value
            if thresh_n[index] > 1:
                thresh_n[index] -= 1
                total_thresh -= 1
            index = (index + 1) % (int(num_thresholds/4) - 1)
            
        thresh = []
        for i, thresh_i in enumerate(thresh_n):
            if thresh_i:
                thresh += np.linspace(
                    equal_space[i],
                    equal_space[i+1], thresh_i+2, endpoint=True
                ).tolist()[1:-1]
        thresh = sorted(thresh)
        thresholds[dp] = thresh

    return thresholds


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def show_stats(summary, keys, cuts=None, show_mean=True):
    # cuts = cuts or [.50, .75, .25]
    # cuts = cuts or [.10, .25, .50, .75, .90]
    # cuts = cuts or [.01, .10, .25, .50, .75, .90, .99]
    cuts = cuts or [.25, .50, .75]
    df = pd.DataFrame(summary)
    # key_max_len = max(len(k) for k in keys)
    key_max_len = 10
    for key in keys:
        quantiles = df.quantile(cuts)[key].to_list()
        print(f'{key.rjust(key_max_len)}:',
              f'({df[key].mean():.4f})' if show_mean else '',
              ','.join(f'{i:.4f}' for i in quantiles))
