import newSimulatorBasedOnJava.simulation.DecisionSituation as DecisionSituation


class RoutingDecisionSituation(DecisionSituation.DecisionSituation):
    def __init__(self, operationOptions, systemState, **kwargs):
        DecisionSituation.DecisionSituation.__init__(self)
        self.queue = operationOptions
        self.systemState = systemState

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
        clonedState = self.systemState.clone()
        return RoutingDecisionSituation(clonedQ, clonedState)