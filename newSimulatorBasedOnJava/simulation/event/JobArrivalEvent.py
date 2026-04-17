from newSimulatorBasedOnJava.simulation.event import AbstractEvent
import newSimulatorBasedOnJava.simulation.HeterogeneousSimulation as HeterogeneousSimulation
import newSimulatorBasedOnJava.jobshop.FirstOperationUploadProcess as FirstOperationUploadProcess
import newSimulatorBasedOnJava.simulation.RoutingDecisionSituation as RoutingDecisionSituation
import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadStartEvent as FirstOperationUploadStartEvent
import newSimulatorBasedOnJava.simulation.event.OperationVisitEvent as OperationVisitEvent


class JobArrivalEvent(AbstractEvent.AbstractEvent):
    def __init__(self, job, **kwargs):
        AbstractEvent.AbstractEvent.__init__(self, job.getArrivalTime())
        self.job = job

    def getName(self):
        return "JobArrivalEvent"

    def trigger(self, simulation):
        operation = self.job.getOperation(0)

        # for speed up by mengxu
        i = 0
        while i < len(operation.getOperationOptions()):
            op = operation.getOperationOptions()[i]
            op.setReadyTime(self.job.getReleaseTime())
            i += 1
        # original
        # for op in operation.getOperationOptions():
        #     op.setReadyTime(self.job.getReleaseTime())

        decisionSituation = RoutingDecisionSituation.RoutingDecisionSituation(operation.getOperationOptions(),
                                                                              simulation.getSystemState())
        operationOption = simulation.getRoutingRule().nextOperationOption(decisionSituation)

        if isinstance(simulation, HeterogeneousSimulation.HeterogeneousSimulation):
            selectedWorkCenterID = operationOption.getWorkCenter().getId()
            uploadTime = simulation.getEntryWorkCenterTranserTime(selectedWorkCenterID)
            operationOption.getOperation().setSelectedWorkCenter(operationOption.getWorkCenter())
            operationOption.getOperation().setUploadTimeToSelectedWorkCenter(uploadTime)
            process = FirstOperationUploadProcess.FirstOperationUploadProcess(operationOption.getWorkCenter(),
                                                                              operationOption,
                                                                              self.job.getReleaseTime())
            simulation.addEvent(FirstOperationUploadStartEvent.FirstOperationUploadStartEvent(process))
        else:
            simulation.addEvent(OperationVisitEvent.OperationVisitEvent(self.job.getReleaseTime(), operationOption))

        # simulation.generateJob() # original
        if simulation.numJobsArrived < simulation.warmupJobs + simulation.getNumJobsRecorded():
            simulation.generateJob()  # stop generate when a number of jobs has been generated

    def addSequencingDecisionSituation(self, simulation, situations, minQueueLength):
        self.trigger(simulation)

    def addRoutingDecisionSituation(self, simulation, situations, minQueueLength):
        operation = self.job.getOperation(0)
        for op in operation.getOperationOptions():
            op.setReadyTime(self.job.getReleaseTime())
        decisionSituation = RoutingDecisionSituation.RoutingDecisionSituation(operation.getOperationOptions(),
                                                                              simulation.getSystemState())
        if operation.getOperationOptions().size() == minQueueLength:
            situations.add(decisionSituation.clone())
        operationOption = simulation.getRoutingRule().nextOperationOption(decisionSituation)

        if isinstance(simulation, HeterogeneousSimulation.HeterogeneousSimulation):
            selectedWorkCenterID = operationOption.getWorkCenter().getId()
            uploadTime = simulation.getEntryWorkCenterTranserTime(selectedWorkCenterID)
            operationOption.getOperation().setSelectedWorkCenter(operationOption.getWorkCenter())
            operationOption.getOperation().setUploadTimeToSelectedWorkCenter(uploadTime)
            process = FirstOperationUploadProcess.FirstOperationUploadProcess(operationOption.getWorkCenter(),
                                                                              operationOption,
                                                                              self.job.getReleaseTime())
            simulation.addEvent(FirstOperationUploadStartEvent.FirstOperationUploadStartEvent(process))
        else:
            simulation.addEvent(OperationVisitEvent.OperationVisitEvent(self.job.getReleaseTime(), operationOption))
        simulation.generateJob()

    def __cmp__(self, other):
        if self.time < other.time:
            return -1
        if self.time > other.time:
            return 1

        if isinstance(other, JobArrivalEvent):
            if self.job.getId() < other.job.getId():
                return -1
            if self.job.getId() > other.job.getId():
                return 1

        return -1
