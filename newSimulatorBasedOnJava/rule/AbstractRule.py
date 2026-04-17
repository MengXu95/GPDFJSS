

class AbstractRule:
    def __init__(self, name, type, **kwargs):
        self.name = name
        self.type = type

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    def priorOperation(self, sequencingDecisionSituation):
        queue = sequencingDecisionSituation.getQueue()
        workCenter = sequencingDecisionSituation.getWorkCenter()
        systemState = sequencingDecisionSituation.getSystemState()
        priorOp = queue[0]
        op_idx = 0
        priorOp.setPriority(self.priority(priorOp, workCenter, systemState))
        for i in range(1, len(queue)):
            op = queue[i]
            op.setPriority(self.priority(op, workCenter, systemState))
            if op.priorTo(priorOp):
                priorOp = op
                op_idx = i
        return priorOp, op_idx

    def nextOperationOption(self, routingDecisionSituation):
        queue = routingDecisionSituation.getQueue()
        systemState = routingDecisionSituation.getSystemState()
        bestOperationOption = queue[0]
        bestOperationOption.setPriority(self.priority(bestOperationOption, bestOperationOption.getWorkCenter(), systemState))
        for i in range(1, len(queue)):
            operationOption = queue[i]
            operationOption.setPriority(self.priority(operationOption, operationOption.getWorkCenter(), systemState))
            # print()
            # print(operationOption.priority)
            # print(bestOperationOption.priority)
            if operationOption.priorTo(bestOperationOption):
                bestOperationOption = operationOption
        return bestOperationOption

    def priority(self, op, workCenter, systemState):
        pass

