import newSimulatorBasedOnJava.simulation.DecisionSituation as DecisionSituation


class SequencingDecisionSituation(DecisionSituation.DecisionSituation):
    def __init__(self, operationOptions, workCenter, systemState, **kwargs):
        DecisionSituation.DecisionSituation.__init__(self)
        self.queue = operationOptions
        self.workCenter = workCenter
        self.systemState = systemState

    def getWorkCenter(self):
        return self.workCenter

    def clone(self):
        clonedQ = []
        # for speed up
        i = 0
        while i < len(self.queue):
            op = self.queue[i]
            clonedQ.append(op.clone())

        # original
        # for op in self.queue:
        #     clonedQ.append(op.clone())
        clonedWC = self.workCenter.clone()
        clonedState = self.systemState.clone()
        return SequencingDecisionSituation(clonedQ, clonedWC, clonedState)