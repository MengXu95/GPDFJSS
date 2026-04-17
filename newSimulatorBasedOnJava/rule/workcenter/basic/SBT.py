from newSimulatorBasedOnJava.rule.AbstractRule import AbstractRule


class SBT(AbstractRule):
    def __init__(self, type, **kwargs):
        name = "SBT"
        AbstractRule.__init__(self, name, type)

    def priority(self, op, workCenter, systemState):
        return workCenter.getBusyTime()