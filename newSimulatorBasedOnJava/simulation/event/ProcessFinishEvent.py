
import newSimulatorBasedOnJava.jobshop.LastOperationDownloadProcess as LastOperationDownloadProcess
import newSimulatorBasedOnJava.jobshop.Process as Process
import newSimulatorBasedOnJava.jobshop.TransportationProcess as TransportationProcess
import newSimulatorBasedOnJava.simulation.HeterogeneousSimulation as HeterogeneousSimulation
import newSimulatorBasedOnJava.simulation.SequencingDecisionSituation as SequencingDecisionSituation
from newSimulatorBasedOnJava.simulation.event import AbstractEvent

import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadFinishEvent as FirstOperationUploadFinishEvent
import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadStartEvent as FirstOperationUploadStartEvent
import newSimulatorBasedOnJava.simulation.event.JobArrivalEvent as JobArrivalEvent
import newSimulatorBasedOnJava.simulation.event.LastOperationDownloadStartEvent as LastOperationDownloadStartEvent
import newSimulatorBasedOnJava.simulation.event.OperationVisitEvent as OperationVisitEvent
import newSimulatorBasedOnJava.simulation.event.ProcessStartEvent as ProcessStartEvent
import newSimulatorBasedOnJava.simulation.event.TransportationStartEvent as TransportationStartEvent


class ProcessFinishEvent(AbstractEvent.AbstractEvent):
    def __init__(self, process, **kwargs):
        AbstractEvent.AbstractEvent.__init__(self, process.getFinishTime())
        self.process = process
        self.op_idx = None # add by mengxu for RL training

    def getName(self):
        return "ProcessFinishEvent"

    def trigger(self,simulation):
        workCenter = self.process.getWorkCenter()
        self.process.getOperationOption().getJob().addProcessFinishEvent(self)
        self.process.getOperationOption().getOperation().completed = True
        if len(workCenter.getQueue()) != 0:
            sequencingDecisionSituation = SequencingDecisionSituation.SequencingDecisionSituation(workCenter.getQueue(), workCenter, simulation.getSystemState())
            dispatchedOp,op_idx = simulation.getSequencingRule().priorOperation(sequencingDecisionSituation)
            workCenter.removeFromQueue(dispatchedOp)
            self.op_idx = op_idx
            processStartTime = max(workCenter.getReadyTime(), self.time)
            nextP = Process.Process(workCenter, self.process.getMachineId(), dispatchedOp, processStartTime)
            simulation.addEvent(ProcessStartEvent.ProcessStartEvent(nextP))

        nextOp= self.process.getOperationOption().getNext(simulation.getSystemState(), simulation.getRoutingRule())
        if nextOp is None:
            if isinstance(simulation, HeterogeneousSimulation.HeterogeneousSimulation):
                workCenterID = self.process.getOperationOption().getWorkCenter().getId()
                downloadTime = simulation.getWorkCenterExitTranserTime(workCenterID)
                self.process.getOperationOption().getOperation().setDownloadTimeToSelectedWorkCenter(downloadTime)
                downloadProcess = LastOperationDownloadProcess.LastOperationDownloadProcess(self.process.getWorkCenter(), self.process.getOperationOption(), self.time)
                simulation.addEvent(LastOperationDownloadStartEvent.LastOperationDownloadStartEvent(downloadProcess))
            else:
                job = self.process.getOperationOption().getJob()
                job.setCompletionTime(self.process.getFinishTime())
                simulation.completeJob(job)
        else:
            if isinstance(simulation, HeterogeneousSimulation.HeterogeneousSimulation):
                nextOp.getOperation().setSelectedWorkCenter(nextOp.getWorkCenter())
                fromWorkCenterID = self.process.getOperationOption().getWorkCenter().getId()
                toWorkCenterID = nextOp.getWorkCenter().getId()
                transferTime = simulation.getWorkCenterTranserTime(fromWorkCenterID,toWorkCenterID)
                nextOp.getOperation().setTransferTimeToSelectedWorkCenter(transferTime)
                transportationProcess = TransportationProcess.TransportationProcess(self.process.getOperationOption().getWorkCenter(), nextOp.getWorkCenter(), nextOp, self.time)
                simulation.addEvent(TransportationStartEvent.TransportationStartEvent(transportationProcess))
            else:
                simulation.addEvent(OperationVisitEvent.OperationVisitEvent(self.time, nextOp))

    def addSequencingDecisionSituation(self, simulation, situations, minQueueLength):
        workCenter = self.process.getWorkCenter()
        self.process.getOperationOption().getJob().addProcessFinishEvent(self)
        if len(workCenter.getQueue()) != 0:
            sequencingDecisionSituation = SequencingDecisionSituation.SequencingDecisionSituation(workCenter.getQueue(), workCenter, simulation.getSystemState())
            if len(workCenter.getQueue()) == minQueueLength:
                situations.add(sequencingDecisionSituation.clone())
            dispatchedOp = simulation.getSequencingRule().priorOperation(sequencingDecisionSituation)
            workCenter.removeFromQueue(dispatchedOp)
            processStartTime = max(workCenter.getReadyTime(), self.time)
            nextP = Process.Process(workCenter, self.process.getMachineId(), dispatchedOp, processStartTime)
            simulation.addEvent(ProcessStartEvent.ProcessStartEvent(nextP))

        nextOp = self.process.getOperationOption().getNext(simulation.getSystemState(), simulation.getRoutingRule())
        if nextOp is None:
            if isinstance(simulation, HeterogeneousSimulation.HeterogeneousSimulation):
                workCenterID = self.process.getOperationOption().getWorkCenter().getId()
                downloadTime = simulation.getWorkCenterExitTranserTime(workCenterID)
                self.process.getOperationOption().getOperation().setDownloadTimeToSelectedWorkCenter(downloadTime)
                downloadProcess = LastOperationDownloadProcess.LastOperationDownloadProcess(self.process.getWorkCenter(), self.process.getOperationOption(), self.time)
                simulation.addEvent(LastOperationDownloadStartEvent.LastOperationDownloadStartEvent(downloadProcess))
            else:
                job = self.process.getOperationOption().getJob()
                job.setCompletionTime(self.process.getFinishTime())
                simulation.completeJob(job)
        else:
            if isinstance(simulation, HeterogeneousSimulation.HeterogeneousSimulation):
                nextOp.getOperation().setSelectedWorkCenter(nextOp.getWorkCenter())
                fromWorkCenterID = self.process.getOperationOption().getWorkCenter().getId()
                toWorkCenterID = nextOp.getWorkCenter().getId()
                transferTime = simulation.getWorkCenterTranserTime(fromWorkCenterID,toWorkCenterID)
                nextOp.getOperation().setTransferTimeToSelectedWorkCenter(transferTime)
                transportationProcess = TransportationProcess.TransportationProcess(self.process.getOperationOption().getWorkCenter(), nextOp.getWorkCenter(), nextOp, self.time)
                simulation.addEvent(TransportationStartEvent.TransportationStartEvent(transportationProcess))
            else:
                simulation.addEvent(OperationVisitEvent.OperationVisitEvent(self.time, nextOp))

    def addRoutingDecisionSituation(self, simulation, situations, minOptions):
        workCenter = self.process.getWorkCenter()
        self.process.getOperationOption().getJob().addProcessFinishEvent(self)
        if len(workCenter.getQueue()) != 0:
            sequencingDecisionSituation = SequencingDecisionSituation.SequencingDecisionSituation(workCenter.getQueue(), workCenter, simulation.getSystemState())
            dispatchedOp = simulation.getSequencingRule().priorOperation(sequencingDecisionSituation)
            workCenter.removeFromQueue(dispatchedOp)
            processStartTime = max(workCenter.getReadyTime(), self.time)
            nextP = Process.Process(workCenter, self.process.getMachineId(), dispatchedOp, processStartTime)
            simulation.addEvent(ProcessStartEvent.ProcessStartEvent(nextP))

        if self.process.getOperationOption().getOperation().getNext() != None:
            if len(self.process.getOperationOption().getOperation().getNext().getOperationOptions()) == minOptions:
                o = self.process.getOperationOption().getOperation()
                r = o.getNext().routingDecisionSituation(simulation.getSystemState())
                situations.add(r.clone())

        nextOp = self.process.getOperationOption().getNext(simulation.getSystemState(), simulation.getRoutingRule())
        if nextOp is None:
            if isinstance(simulation, HeterogeneousSimulation.HeterogeneousSimulation):
                workCenterID = self.process.getOperationOption().getWorkCenter().getId()
                downloadTime = simulation.getWorkCenterExitTranserTime(workCenterID)
                self.process.getOperationOption().getOperation().setDownloadTimeToSelectedWorkCenter(downloadTime)
                downloadProcess = LastOperationDownloadProcess.LastOperationDownloadProcess(self.process.getWorkCenter(),
                                                               self.process.getOperationOption(), self.time)
                simulation.addEvent(LastOperationDownloadStartEvent.LastOperationDownloadStartEvent(downloadProcess))
            else:
                job = self.process.getOperationOption().getJob()
                job.setCompletionTime(self.process.getFinishTime())
                simulation.completeJob(job)
        else:
            if isinstance(simulation, HeterogeneousSimulation.HeterogeneousSimulation):
                nextOp.getOperation().setSelectedWorkCenter(nextOp.getWorkCenter())
                fromWorkCenterID = self.process.getOperationOption().getWorkCenter().getId()
                toWorkCenterID = nextOp.getWorkCenter().getId()
                transferTime = simulation.getWorkCenterTranserTime(fromWorkCenterID, toWorkCenterID)
                nextOp.getOperation().setTransferTimeToSelectedWorkCenter(transferTime)
                transportationProcess = TransportationProcess.TransportationProcess(self.process.getOperationOption().getWorkCenter(),
                                                              nextOp.getWorkCenter(), nextOp, self.time)
                simulation.addEvent(TransportationStartEvent.TransportationStartEvent(transportationProcess))
            else:
                simulation.addEvent(OperationVisitEvent.OperationVisitEvent(self.time, nextOp))

    def __cmp__(self, other):
        if self.time < other.time:
            return -1
        if self.time > other.time:
            return 1

        if isinstance(other, ProcessFinishEvent):
            if self.process.getWorkCenter().getId() < other.process.getWorkCenter().getId():
                return -1
            if self.process.getWorkCenter().getId() > other.process.getWorkCenter().getId():
                return 1

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