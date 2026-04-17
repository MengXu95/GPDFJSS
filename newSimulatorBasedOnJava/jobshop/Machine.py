


class Machine:
    def __init__(self, id, workCenter, readyTime, **kwargs):
        self.id = id
        self.workCenter = workCenter
        self.readyTime = readyTime

    def getId(self):
        return self.id

    def getWorkCenter(self):
        return self.workCenter

    def getReadyTime(self):
        return self.readyTime

    def setReadyTime(self, readyTime):
        self.readyTime = readyTime

    def equals(self, other):
        return self.id == other.id