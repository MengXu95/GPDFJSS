import newSimulatorBasedOnJava.rule.AbstractRule as AbstractRule


class WINQ(AbstractRule.AbstractRule):
    def __init__(self, type, **kwargs):
        name = "WINQ"
        AbstractRule.AbstractRule.__init__(self, name, type)

    def priority(self, op, workCenter, systemState):
        return op.getOperation().getAverageWIQForRemainingOperations()