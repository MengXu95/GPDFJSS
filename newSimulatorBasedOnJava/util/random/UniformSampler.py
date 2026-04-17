from newSimulatorBasedOnJava.util.random.AbstractRealSampler import AbstractRealSampler


class UniformSampler(AbstractRealSampler):
    def __init__(self, lower, upper, **kwargs):
        AbstractRealSampler.__init__(self)
        self.lower = lower
        self.upper = upper

    def next(self, rdg):
        # return rdg.nextUniform(self.lower, self.upper)
        return rdg.uniform(self.lower, self.upper)

    def set(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def setLower(self, lower):
        self.lower = lower

    def setUpper(self, upper):
        self.upper = upper

    def getLower(self):
        return self.lower

    def getUpper(self):
        return self.upper

    def setMean(self, mean):
        pass

    def getMean(self):
        return (self.lower + self.upper) * 0.5

    def clone(self):
        return UniformSampler(self.lower, self.upper)