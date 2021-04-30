"""
@Author: P_k_y
@Time: 2021/4/12
"""
import functools
import time


def time_log(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print('\033[0;33m[Logger Info] Function "%s()" has used %fms.\033[0m' % (func.__name__, (end - start) * 1000))
        return res

    return wrapper


@time_log
def task(a, b):
    time.sleep(0.8)
    return a+b


if __name__ == '__main__':
    task(2, 3)

