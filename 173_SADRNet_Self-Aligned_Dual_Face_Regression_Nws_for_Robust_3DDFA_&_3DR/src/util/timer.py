import time


class Timer:
    def __init__(self):
        self.last_time = 0

    def begin(self):
        self.last_time = time.time()

    def mark(self, s=''):
        self.timer_count(s)
        self.last_time = time.time()

    def timer_count(self, s=''):
        print(s, time.time() - self.last_time)


TIMER = Timer()
