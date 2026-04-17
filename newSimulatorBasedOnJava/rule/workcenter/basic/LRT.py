from newSimulatorBasedOnJava.rule.AbstractRule import AbstractRule


class LRT(AbstractRule):
    def __init__(self, type, **kwargs):
        name = "LRT"
        AbstractRule.__init__(self, name, type)

    def priority(self, op, workCenter, systemState):
        return -workCenter.getReadyTime()