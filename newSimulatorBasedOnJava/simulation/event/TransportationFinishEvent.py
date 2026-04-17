from newSimulatorBasedOnJava.simulation.event import AbstractEvent
import newSimulatorBasedOnJava.simulation.event.OperationVisitEvent as OperationVisitEvent
import newSimulatorBasedOnJava.simulation.event.TransportationStartEvent as TransportationStartEvent
import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadStartEvent as FirstOperationUploadStartEvent
import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadFinishEvent as FirstOperationUploadFinishEvent
import newSimulatorBasedOnJava.simulation.event.JobArrivalEvent as JobArrivalEvent
import newSimulatorBasedOnJava.simulation.event.LastOperationDownloadFinishEvent as LastOperationDownloadFinishEvent
import newSimulatorBasedOnJava.simulation.event.LastOperationDownloadStartEvent as LastOperationDownloadStartEvent
import newSimulatorBasedOnJava.simulation.event.ProcessFinishEvent as ProcessFinishEvent
import newSimulatorBasedOnJava.simulation.event.ProcessStartEvent as ProcessStartEvent


class TransportationFinishEvent(AbstractEvent.AbstractEvent):
    def __init__(self, process, **kwargs):
        AbstractEvent.AbstractEvent.__init__(self, process.getFinishTime())
        self.process = process

    def getName(self):
        return "TransportationFinishEvent"

    def trigger(self,simulation):
        operationOption = self.process.getOperationOption()
        operationOption.setReadyTime(self.time)
        simulation.addEvent(OperationVisitEvent.OperationVisitEvent(self.time, operationOption))

    def addSequencingDecisionSituation(self, simulation, situations, minQueueLength):
        self.trigger(simulation)

    def addRoutingDecisionSituation(self, simulation, situations, minOptions):
        self.trigger(simulation)

    def __cmp__(self, other):
        if self.time < other.time:
            return -1
        if self.time > other.time:
            return 1

        if isinstance(other, TransportationStartEvent.TransportationStartEvent):
            return 1
        if isinstance(other, FirstOperationUploadStartEvent.FirstOperationUploadStartEvent):
            return 1
        if isinstance(other, LastOperationDownloadStartEvent.LastOperationDownloadStartEvent):
            return 1
        if isinstance(other, TransportationFinishEvent):
            return 0
        if isinstance(other, FirstOperationUploadFinishEvent.FirstOperationUploadFinishEvent):
            return 0
        if isinstance(other, LastOperationDownloadFinishEvent.LastOperationDownloadFinishEvent):
            return 0
        if isinstance(other, JobArrivalEvent.JobArrivalEvent):
            return 1
        if isinstance(other, OperationVisitEvent.OperationVisitEvent):
            return 1
        if isinstance(other, ProcessStartEvent.ProcessStartEvent):
            return 1
        if isinstance(other, ProcessFinishEvent.ProcessFinishEvent):
            return 1

        return -1

    def getProcess(self):
        return self.process