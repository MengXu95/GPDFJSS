from newSimulatorBasedOnJava.simulation.event import AbstractEvent
import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadFinishEvent as FirstOperationUploadFinishEvent
import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadStartEvent as FirstOperationUploadStartEvent
import newSimulatorBasedOnJava.simulation.event.JobArrivalEvent as JobArrivalEvent
import newSimulatorBasedOnJava.simulation.event.LastOperationDownloadFinishEvent as LastOperationDownloadFinishEvent
import newSimulatorBasedOnJava.simulation.event.LastOperationDownloadStartEvent as LastOperationDownloadStartEvent
import newSimulatorBasedOnJava.simulation.event.OperationVisitEvent as OperationVisitEvent
import newSimulatorBasedOnJava.simulation.event.ProcessFinishEvent as ProcessFinishEvent
import newSimulatorBasedOnJava.simulation.event.ProcessStartEvent as ProcessStartEvent
import newSimulatorBasedOnJava.simulation.event.TransportationFinishEvent as TransportationFinishEvent


class TransportationStartEvent(AbstractEvent.AbstractEvent):
    def __init__(self, process, **kwargs):
        AbstractEvent.AbstractEvent.__init__(self, process.getStartTime())
        self.process = process

    def getName(self):
        return "TransportationStartEvent"

    def trigger(self,simulation):
        simulation.addEvent(TransportationFinishEvent.TransportationFinishEvent(self.process))

    def addSequencingDecisionSituation(self, simulation, situations, minQueueLength):
        self.trigger(simulation)

    def addRoutingDecisionSituation(self, simulation, situations, minOptions):
        self.trigger(simulation)

    def __cmp__(self, other):
        if self.time < other.time:
            return -1
        if self.time > other.time:
            return 1

        if isinstance(other, TransportationStartEvent):
            return 0
        if isinstance(other, FirstOperationUploadStartEvent.FirstOperationUploadStartEvent):
            return 0
        if isinstance(other, LastOperationDownloadStartEvent.LastOperationDownloadStartEvent):
            return 0
        if isinstance(other, TransportationFinishEvent.TransportationFinishEvent):
            return -1
        if isinstance(other, FirstOperationUploadFinishEvent.FirstOperationUploadFinishEvent):
            return -1
        if isinstance(other, LastOperationDownloadFinishEvent.LastOperationDownloadFinishEvent):
            return -1
        if isinstance(other, JobArrivalEvent.JobArrivalEvent):
            return 1
        if isinstance(other, OperationVisitEvent.OperationVisitEvent):
            return -1
        if isinstance(other, ProcessStartEvent.ProcessStartEvent):
            return 1
        if isinstance(other, ProcessFinishEvent.ProcessFinishEvent):
            return 1

        return -1

    def getProcess(self):
        return self.process