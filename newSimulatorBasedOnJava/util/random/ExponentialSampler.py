from newSimulatorBasedOnJava.util.random.AbstractRealSampler import AbstractRealSampler


class ExponentialSampler(AbstractRealSampler):
    def __init__(self, mean, **kwargs):
        AbstractRealSampler.__init__(self)
        self.mean = mean

    def next(self, rdg):
        # return rdg.nextExponential(self.mean)
        return rdg.exponential(self.mean)

    def setLower(self, lower):
        pass

    def setUpper(self, upper):
        pass

    def setMean(self, mean):
        self.mean = mean

    def getMean(self):
        return self.mean

    def clone(self):
        return ExponentialSampler(self.mean)