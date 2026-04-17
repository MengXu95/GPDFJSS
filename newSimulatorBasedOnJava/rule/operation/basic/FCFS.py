import newSimulatorBasedOnJava.rule.AbstractRule as AbstractRule


class FCFS(AbstractRule.AbstractRule):
    def __init__(self, type, **kwargs):
        name = "FCFS"
        AbstractRule.AbstractRule.__init__(self, name, type)

    def priority(self, op, workCenter, systemState):
        return op.getReadyTime()