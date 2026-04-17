from newSimulatorBasedOnJava.rule.AbstractRule import AbstractRule


class SRT(AbstractRule):
    def __init__(self, type, **kwargs):
        name = "SRT"
        AbstractRule.__init__(self, name, type)

    def priority(self, op, workCenter, systemState):
        return workCenter.getReadyTime()