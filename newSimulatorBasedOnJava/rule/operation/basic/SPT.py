import newSimulatorBasedOnJava.rule.AbstractRule as AbstractRule


class SPT(AbstractRule.AbstractRule):
    def __init__(self, type, **kwargs):
        name = "SPT"
        AbstractRule.AbstractRule.__init__(self, name, type)

    def priority(self, op, workCenter, systemState):
        return op.getProcTime()