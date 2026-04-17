import newSimulatorBasedOnJava.rule.AbstractRule as AbstractRule


class CR(AbstractRule.AbstractRule):
    def __init__(self, type, **kwargs):
        name = "CR"
        AbstractRule.AbstractRule.__init__(self, name, type)

    def priority(self, op, workCenter, systemState):
        return (op.getJob().getDueDate() - systemState.getClockTime())/(op.getProcTime() + op.getWorkRemaining())