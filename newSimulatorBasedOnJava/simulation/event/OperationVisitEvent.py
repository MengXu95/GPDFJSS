import newSimulatorBasedOnJava.jobshop.Process as Process
from newSimulatorBasedOnJava.simulation.event import AbstractEvent
import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadFinishEvent as FirstOperationUploadFinishEvent
import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadStartEvent as FirstOperationUploadStartEvent
import newSimulatorBasedOnJava.simulation.event.JobArrivalEvent as JobArrivalEvent
import newSimulatorBasedOnJava.simulation.event.ProcessStartEvent as ProcessStartEvent


class OperationVisitEvent(AbstractEvent.AbstractEvent):
    def __init__(self, time, operationOption, **kwargs):
        AbstractEvent.AbstractEvent.__init__(self, time)
        self.operationOption = operationOption

    def getName(self):
        return "OperationVisitEvent"

    def trigger(self,simulation):
        self.operationOption.setReadyTime(self.time)
        workCenter = self.operationOption.getWorkCenter()
        earliestMachine = workCenter.earliestReadyMachine()
        p = Process.Process(workCenter, earliestMachine.getId(), self.operationOption, self.time)

        if earliestMachine.getReadyTime() > self.time or not simulation.canAddToQueue(p) or workCenter.numOpsInQueue() >= 1:
            workCenter.addToQueue(self.operationOption)
        else:
            processStartEvent = ProcessStartEvent.ProcessStartEvent(p)
            simulation.addEvent(processStartEvent)


    def addSequencingDecisionSituation(self, simulation, situations, minQueueLength):
        self.trigger(simulation)

    def addRoutingDecisionSituation(self, simulation, situations, minQueueLength):
        self.trigger(simulation)

    def __cmp__(self, other):
        if self.time < other.time:
            return -1
        if self.time > other.time:
            return 1

        if isinstance(other, JobArrivalEvent.JobArrivalEvent):
            return 1
        if isinstance(other, OperationVisitEvent):
            return 0
        if isinstance(other, FirstOperationUploadStartEvent.FirstOperationUploadStartEvent):
            return 1
        if isinstance(other, FirstOperationUploadFinishEvent.FirstOperationUploadFinishEvent):
            return 1

        return -1