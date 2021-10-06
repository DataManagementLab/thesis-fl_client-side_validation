from enum import unique
import time
from collections import defaultdict
from typing import Optional

class TimeTracker:

    def __init__(self):
        self.clear()

    def start(self, id):
        assert not id in self.start_times, f"Timer of id '{id}' is already running."
        self.start_times[id] = time.time()

    def stop(self, id):
        assert id in self.start_times, f"Timer of id '{id}' is not running and can not be stopped."
        time_diff = time.time() - self.start_times[id]
        self.total_times[id] += time_diff
        self.total_times_history[id].append(time_diff)
        del self.start_times[id]

    def reset(self, id):
        assert not id in self.start_times, f"Timer of id '{id}' can not be reset while it is running."
        if id in self.total_times:
            del self.total_times[id]
            del self.total_times_history[id]
    
    def clear(self):
        self.total_times = defaultdict(float)
        self.total_times_history = defaultdict(list)
        self.start_times = dict()
    
    def last(self, id, n=1, get_range=False):
        assert n <= len(self.total_times_history[id]), f"There are less than {n} elements in history for id {id}."
        if get_range:
            return self.total_times_history[id][-n:]
        else:
            return self.total_times_history[id][-n]
    
    def get(self, id, default=None):
        if self.has(id):
            return self[id]
        else:
            return default
    
    def has(self, id):
        return id in self.total_times or id in self.start_times
    
    @property
    def ids(self):
        return set(self.total_times) | set(self.start_times)
    
    def __getitem__(self, id):
        assert self.has(id), f"Timer of id '{id}' is unknown."
        res = 0.0
        if id in self.total_times: res += self.total_times[id]
        if id in self.start_times: res += time.time() - self.start_times[id]
        return res

    def __str__(self):
        res = ""
        for n, id in enumerate(sorted(self.ids)):
            res += "{}\t{}\t{}".format(id, self[id], "(Running)" if id in self.start_times else "") 
            if n + 1 < len(self.ids): res += "\n"
        return res

    @classmethod
    def from_dict(cls, time_dict):
        assert 'total_times' in time_dict and 'total_times_history' in time_dict and 'start_times' in time_dict, "Invalid time dictionary"
        tt = cls()
        tt.total_times = defaultdict(float, time_dict['total_times'])
        tt.total_times_history = defaultdict(list, time_dict['total_times_history'])
        tt.start_times = time_dict['start_times']
        return tt

    def get_dict(self):
        return dict(
            total_times=self.total_times,
            total_times_history=self.total_times_history,
            start_times=self.start_times)



