

class AbstractRealSampler:
    def __init__(self, **kwargs):
        pass

    def next(self, rdg):
        pass

    def setLower(self, lower):
        pass

    def setUpper(self, upper):
        pass

    def setMean(self, mean):
        pass

    def getMean(self):
        pass

    def clone(self):
        pass