from newSimulatorBasedOnJava.simulation.event import AbstractEvent
import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadFinishEvent as FirstOperationUploadFinishEvent
import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadStartEvent as FirstOperationUploadStartEvent
import newSimulatorBasedOnJava.simulation.event.JobArrivalEvent as JobArrivalEvent
import newSimulatorBasedOnJava.simulation.event.OperationVisitEvent as OperationVisitEvent
import newSimulatorBasedOnJava.simulation.event.ProcessFinishEvent as ProcessFinishEvent


class ProcessStartEvent(AbstractEvent.AbstractEvent):
    def __init__(self, process, **kwargs):
        AbstractEvent.AbstractEvent.__init__(self, process.getStartTime())
        self.process = process

    def getName(self):
        return "ProcessStartEvent"

    def trigger(self,simulation):
        workCenter = self.process.getWorkCenter()
        workCenter.setMachineReadyTime(self.process.getMachineId(), self.process.getFinishTime())
        workCenter.incrementBusyTime(self.process.getDuration())

        # simulation.addEvent(ProcessFinishEvent.ProcessFinishEvent(self.process.getFinishTime(), self.process))
        simulation.addEvent(ProcessFinishEvent.ProcessFinishEvent(self.process))

    def addSequencingDecisionSituation(self, simulation, situations, minQueueLength):
        self.trigger(simulation)

    def addRoutingDecisionSituation(self, simulation, situations, minOptions):
        self.trigger(simulation)

    def __cmp__(self, other):
        if self.time < other.time:
            return -1
        if self.time > other.time:
            return 1

        if isinstance(other, ProcessStartEvent):
            return 0
        if isinstance(other, ProcessFinishEvent.ProcessFinishEvent):
            return -1
        if isinstance(other, JobArrivalEvent.JobArrivalEvent):
            return 1
        if isinstance(other, OperationVisitEvent.OperationVisitEvent):
            return 1
        if isinstance(other, FirstOperationUploadStartEvent.FirstOperationUploadStartEvent):
            return 1
        if isinstance(other, FirstOperationUploadFinishEvent.FirstOperationUploadFinishEvent):
            return 1

        return -1

    def getProcess(self):
        return self.process