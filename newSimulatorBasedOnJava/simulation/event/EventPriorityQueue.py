from heapq import heappush
from heapq import heappop

class EventPriorityQueue(object):
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heappush(self._queue, (priority, self._index, item))
        # heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heappop(self._queue)[-1]

    def qsize(self):
        return len(self._queue)

    def empty(self):
        if len(self._queue) == 0:
            return True
        else:
            return False

    def clear(self):
        self._queue.clear()
        self._index = 0

    # def get(self):
    #     heapq.