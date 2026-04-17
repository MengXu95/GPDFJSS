from newSimulatorBasedOnJava.util.random.AbstractRealSampler import AbstractRealSampler


class TwoSixTwoSampler(AbstractRealSampler):
    def __init__(self, **kwargs):
        AbstractRealSampler.__init__(self)

    def next(self, rdg):
        value = 1
        r = rdg.random()
        if r < 0.2:
            value = 4
        elif r < 0.8:
            value = 2
        return value

    def setLower(self, lower):
        pass

    def setUpper(self, upper):
        pass

    def setMean(self, mean):
        pass

    def getMean(self):
        pass

    def clone(self):
        return TwoSixTwoSampler()