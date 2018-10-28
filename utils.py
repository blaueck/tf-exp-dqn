import time


class TimeMeter:

    def __init__(self):
        self.start_time, self.duration, self.counter = 0., 0., 0.

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.duration += time.perf_counter() - self.start_time
        self.counter += 1

    def get(self):
        return self.duration / self.counter

    def reset(self):
        self.start_time, self.duration, self.counter = 0., 0., 0.


def to_bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if (value == 'true' or value == '1'
        or value == 'yes' or value == 'y'):
        return True
    else:
        return False