from newSimulatorBasedOnJava.rule.AbstractRule import AbstractRule


class WIQ(AbstractRule):
    def __init__(self, type, **kwargs):
        name = "WIQ"
        AbstractRule.__init__(self, name, type)

    def priority(self, op, workCenter, systemState):
        return workCenter.getWorkInQueue()