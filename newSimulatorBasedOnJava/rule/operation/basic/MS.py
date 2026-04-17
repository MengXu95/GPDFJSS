import newSimulatorBasedOnJava.rule.AbstractRule as AbstractRule

'''
Minimum Slack 
'''
class MS(AbstractRule.AbstractRule):
    def __init__(self, type, **kwargs):
        name = "MS"
        AbstractRule.AbstractRule.__init__(self, name, type)

    def priority(self, op, workCenter, systemState):
        return op.getJob().getDueDate() - systemState.getClockTime() - op.getWorkRemaining()