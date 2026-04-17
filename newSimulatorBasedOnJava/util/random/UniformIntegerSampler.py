from newSimulatorBasedOnJava.util.random.AbstractIntegerSampler import AbstractIntegerSampler


class UniformIntegerSampler(AbstractIntegerSampler):
    def __init__(self, lower, upper, **kwargs):
        AbstractIntegerSampler.__init__(self)
        self.lower = lower
        self.upper = upper

    def next(self, rdg):
        return rdg.integers(self.lower, self.upper)

    def set(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def setLower(self, lower):
        self.lower = lower

    def setUpper(self, upper):
        self.upper = upper

    def setMean(self, mean):
        pass

    def getMean(self):
        return (self.lower + self.upper) * 0.5

    def clone(self):
        return UniformIntegerSampler(self.lower, self.upper)