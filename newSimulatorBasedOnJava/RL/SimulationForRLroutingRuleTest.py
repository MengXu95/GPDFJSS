import random

import newSimulatorBasedOnJava.simulation.HeterogeneousSimulation as HeterogeneousSimulation
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from tabulate import tabulate

import newSimulatorBasedOnJava.simulation.event.ProcessFinishEvent as ProcessFinishEvent
import newSimulatorBasedOnJava.rule.workcenter.basic.LRT as LRT
import newSimulatorBasedOnJava.rule.workcenter.basic.SBT as SBT
import newSimulatorBasedOnJava.rule.workcenter.basic.SRT as SRT
import newSimulatorBasedOnJava.rule.workcenter.basic.WIQ as WIQ
import newSimulatorBasedOnJava.simulation.RoutingDecisionSituation as RoutingDecisionSituation
import newSimulatorBasedOnJava.simulation.SequencingDecisionSituation as SequencingDecisionSituation
import newSimulatorBasedOnJava.jobshop.LastOperationDownloadProcess as LastOperationDownloadProcess
import newSimulatorBasedOnJava.jobshop.Process as Process
import newSimulatorBasedOnJava.jobshop.TransportationProcess as TransportationProcess

import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadFinishEvent as FirstOperationUploadFinishEvent
import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadStartEvent as FirstOperationUploadStartEvent
import newSimulatorBasedOnJava.simulation.event.JobArrivalEvent as JobArrivalEvent
import newSimulatorBasedOnJava.simulation.event.LastOperationDownloadStartEvent as LastOperationDownloadStartEvent
import newSimulatorBasedOnJava.simulation.event.LastOperationDownloadFinishEvent as LastOperationDownloadFinishEvent
import newSimulatorBasedOnJava.simulation.event.OperationVisitEvent as OperationVisitEvent
import newSimulatorBasedOnJava.simulation.event.ProcessStartEvent as ProcessStartEvent
import newSimulatorBasedOnJava.simulation.event.TransportationStartEvent as TransportationStartEvent
import newSimulatorBasedOnJava.rule.RuleType as RuleType
import newSimulatorBasedOnJava.jobshop.FirstOperationUploadProcess as FirstOperationUploadProcess
import newSimulatorBasedOnJava.RL.routingRuleTrain.SimulationForRoutingRuleTrain as SimulationForRoutingRuleTrain


class SimulationForRLroutingRuleTest(SimulationForRoutingRuleTrain.SimulationForRoutingRuleTrain):
    def __init__(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, minNumOperations, maxNumOperations,
                utilLevel, dueDateFactor, revisit, **kwargs):
        SimulationForRoutingRuleTrain.SimulationForRoutingRuleTrain.__init__(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, minNumOperations, maxNumOperations,
                utilLevel, dueDateFactor, revisit, sequeningTest = False, routingTest = True, run_seed = kwargs['run_seed'], objective = kwargs['objective'])

        self.bad_run_check = True

        if 'bad_run_check' in kwargs:
            self.bad_run_check = kwargs['bad_run_check']
        # the following is used for RL test
        # each item is a network
        self.net_dict = {}
        self.func_list = [LRT.LRT(RuleType.RuleType.ROUTING), SBT.SBT(RuleType.RuleType.ROUTING),
                          SRT.SRT(RuleType.RuleType.ROUTING), WIQ.WIQ(RuleType.RuleType.ROUTING)]
        self.input_size = self.numWorkCenters * 3 + 3
        self.output_size = len(self.func_list)
        self.path = sys.path[0]
        self.action_NN = None
        if len(self.func_list) == 4:
            self.action_NN = build_network_large(self.input_size, self.output_size)
            self.address_seed = "{}/routing_models/scenario_" + self.dataset_name + "/run_" + str(
                seed) + "_large_state_dict" + '{}wc{}m'.format(self.input_size, self.output_size)
        self.build_state = self.state_deeper
        self.action_NN.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
        self.action_NN.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
        print("---> VALIDATION mode ON <---")
        # processes
        self.setup()


    def run(self):
        self.count = 0
        while not self.eventQueue.empty() and self.throughput < self.numJobsRecorded:
            nextEvent = self.eventQueue.pop()
            self.beforeThroughput = self.throughput
            self.systemState.setClockTime(nextEvent.getTime())
            if isinstance(nextEvent, ProcessFinishEvent.ProcessFinishEvent):
                self.triggerProcessFinishEvent(nextEvent)
            elif isinstance(nextEvent, JobArrivalEvent.JobArrivalEvent):
                self.triggerJobArrivalEvent(nextEvent)
                self.generateJob()
            else:
                nextEvent.trigger(self)
            self.afterThroughput = self.throughput
            if self.bad_run_check:
                if self.throughput > self.warmupJobs and self.afterThroughput - self.beforeThroughput == 0:
                    self.count = self.count + 1
                if self.count > 100000:
                    self.count = 0
                    self.systemState.setClockTime(np.Infinity)
                    print("bad run: self.count > 100000")
                    self.eventQueue.clear()
                for workCenter in self.systemState.getWorkCenters():
                    numOpsInQueue = workCenter.numOpsInQueue()
                    if numOpsInQueue > 100:
                        print("Number warm-up job completed: " + str(len(self.systemState.warmupJobsCompleted)))
                        print("Number job completed: " + str(len(self.systemState.getJobsCompleted())))
                        self.systemState.setClockTime(np.Infinity)
                        self.eventQueue.clear()
                        print("bad run: numOpsInQueue > 100")

    def triggerProcessFinishEvent(self, processFinish):
        workCenter = processFinish.process.getWorkCenter()
        estimated_job_slack = processFinish.process.getOperationOption().getJob().getSlack(self.systemState.getClockTime())
        processFinish.process.getOperationOption().getJob().addProcessFinishEvent(self)
        if len(workCenter.getQueue()) != 0:
            sequencingDecisionSituation = SequencingDecisionSituation.SequencingDecisionSituation(workCenter.getQueue(),
                                                                                                  workCenter,
                                                                                                  self.getSystemState())
            dispatchedOp, op_idx = self.getSequencingRule().priorOperation(sequencingDecisionSituation)
            workCenter.removeFromQueue(dispatchedOp)
            processStartTime = np.maximum(workCenter.getReadyTime(), processFinish.time)
            nextP = Process.Process(workCenter, processFinish.process.getMachineId(), dispatchedOp, processStartTime)
            start_event=ProcessStartEvent.ProcessStartEvent(nextP)
            self.addEvent(start_event)

        nextOperation = processFinish.process.getOperationOption().operation.getNext()

        if nextOperation is None:
            workCenterID = processFinish.process.getOperationOption().getWorkCenter().getId()
            downloadTime = self.getWorkCenterExitTranserTime(workCenterID)
            processFinish.process.getOperationOption().getOperation().setDownloadTimeToSelectedWorkCenter(downloadTime)
            downloadProcess = LastOperationDownloadProcess.LastOperationDownloadProcess(processFinish.process.getWorkCenter(), processFinish.process.getOperationOption(), processFinish.time)
            self.addEvent(LastOperationDownloadStartEvent.LastOperationDownloadStartEvent(downloadProcess))
        else:
            job_idx = nextOperation.getJob().getId()
            job_pt = [-1 for i in range(self.numWorkCenters)]
            for opeOption in nextOperation.getOperationOptions():
                wc_id = opeOption.getWorkCenter().getId()
                job_pt[wc_id] = opeOption.getProcTime()
            job_pt = np.array(job_pt)
            job_slack = nextOperation.getJob().getSlack(self.systemState.getClockTime())
            routing_data = self.routing_data_generation(nextOperation)
            idx_rout = self.action_by_DRL(job_idx, routing_data, job_pt, job_slack, 0, nextOperation)
            selectedRoutingRule = self.func_list[idx_rout]
            self.setRoutingRule(selectedRoutingRule)
            nextOp= processFinish.process.getOperationOption().getNext(self.getSystemState(),self.getRoutingRule())
            nextOp.getOperation().setSelectedWorkCenter(nextOp.getWorkCenter())
            fromWorkCenterID = processFinish.process.getOperationOption().getWorkCenter().getId()
            toWorkCenterID = nextOp.getWorkCenter().getId()
            transferTime = self.getWorkCenterTranserTime(fromWorkCenterID, toWorkCenterID)
            nextOp.getOperation().setTransferTimeToSelectedWorkCenter(transferTime)
            transportationProcess = TransportationProcess.TransportationProcess(
                processFinish.process.getOperationOption().getWorkCenter(), nextOp.getWorkCenter(), nextOp,
                processFinish.time)
            self.addEvent(TransportationStartEvent.TransportationStartEvent(transportationProcess))

    def triggerJobArrivalEvent(self, jobArrival):
        operation = jobArrival.job.getOperation(0)
        for op in operation.getOperationOptions():
            op.setReadyTime(jobArrival.job.getReleaseTime())
        routingDecisionSituation = RoutingDecisionSituation.RoutingDecisionSituation(operation.getOperationOptions(), self.getSystemState())

        job_idx = operation.getJob().getId()
        job_pt = [-1 for i in range(self.numWorkCenters)]
        for opeOption in operation.getOperationOptions():
            wc_id = opeOption.getWorkCenter().getId()
            job_pt[wc_id] = opeOption.getProcTime()
            # job_pt.append(opeOption.getProcTime())
        job_pt = np.array(job_pt)
        job_slack = jobArrival.job.getSlack(self.systemState.getClockTime())
        routing_data = self.routing_data_generation(operation)
        idx_rout= self.action_by_DRL(job_idx, routing_data, job_pt, job_slack, 0, operation)
        selectedRoutingRule = self.func_list[idx_rout]
        self.setRoutingRule(selectedRoutingRule)
        operationOption= self.getRoutingRule().nextOperationOption(routingDecisionSituation)
        selectedWorkCenterID = operationOption.getWorkCenter().getId()
        uploadTime = self.getEntryWorkCenterTranserTime(selectedWorkCenterID)
        operationOption.getOperation().setSelectedWorkCenter(operationOption.getWorkCenter())
        operationOption.getOperation().setUploadTimeToSelectedWorkCenter(uploadTime)
        process = FirstOperationUploadProcess.FirstOperationUploadProcess(operationOption.getWorkCenter(),
                                                                          operationOption,
                                                                          jobArrival.job.getReleaseTime())
        self.addEvent(FirstOperationUploadStartEvent.FirstOperationUploadStartEvent(process))



    def action_by_DRL(self, job_idx, routing_data, job_pt, job_slack, wc_idx, operation, *args):
        s_t = self.build_state(routing_data, job_pt, job_slack, operation)
        # input state to policy network, produce the state-action value
        value = self.action_NN.forward(s_t.reshape(1,1,self.input_size),wc_idx)
        # generate the action
        a_t = torch.argmax(value)
        #print(value,a_t)
        #print('Policy NN choose action')
        return a_t


'''
Neural Networks
'''
class build_network_small(nn.Module):
    def __init__(self, input_size, output_size):
        super(build_network_small, self).__init__()
        # size of layers
        layer_1 = 16
        layer_2 = 16
        layer_3 = 16
        layer_4 = 8
        layer_5 = 8
        # FCNN
        self.fc1 = nn.Linear(input_size, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, layer_3)
        self.fc4 = nn.Linear(layer_3, layer_4)
        self.fc5 = nn.Linear(layer_4, output_size)
        #self.fc6 = nn.Linear(layer_5, output_size)
        # activation functions
        self.tanh = nn.Tanh()
        self.instancenorm = nn.InstanceNorm1d(input_size)
        self.flatten = nn.Flatten()
        # Huber loss function
        self.loss_func = F.smooth_l1_loss

    def forward(self, x, *args):
        #print('original',x)
        x = self.instancenorm(x)
        #print('normalized',x)
        x = self.flatten(x)
        #print('flattened',x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = self.fc5(x)
        #x = self.tanh(x)
        #x = self.fc6(x)
        return x

class build_network_medium(nn.Module):
    def __init__(self, input_size, output_size):
        super(build_network_medium, self).__init__()
        # size of layers
        layer_1 = 32
        layer_2 = 32
        layer_3 = 16
        layer_4 = 8
        layer_5 = 8
        # FCNN
        self.fc1 = nn.Linear(input_size, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, layer_3)
        self.fc4 = nn.Linear(layer_3, layer_4)
        self.fc5 = nn.Linear(layer_4, output_size)
        #self.fc6 = nn.Linear(layer_5, output_size)
        # activation functions
        self.tanh = nn.Tanh()
        self.instancenorm = nn.InstanceNorm1d(input_size)
        self.flatten = nn.Flatten()
        # Huber loss function
        self.loss_func = F.smooth_l1_loss

    def forward(self, x, *args):
        #print('original',x)
        x = self.instancenorm(x)
        #print('normalized',x)
        x = self.flatten(x)
        #print('flattened',x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = self.fc5(x)
        #x = self.tanh(x)
        #x = self.fc6(x)
        return x

class build_network_large(nn.Module):
    def __init__(self, input_size, output_size):
        super(build_network_large, self).__init__()
        # size of layers
        layer_1 = 64
        layer_2 = 64
        layer_3 = 48
        layer_4 = 32
        layer_5 = 16
        layer_6 = 16
        # FCNN
        self.fc1 = nn.Linear(input_size, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, layer_3)
        self.fc4 = nn.Linear(layer_3, layer_4)
        self.fc5 = nn.Linear(layer_4, layer_5)
        self.fc6 = nn.Linear(layer_5, layer_6)
        self.fc7 = nn.Linear(layer_6, output_size)
        # activation functions
        self.tanh = nn.Tanh()
        self.instancenorm = nn.InstanceNorm1d(input_size)
        self.flatten = nn.Flatten()
        # Huber loss function
        self.loss_func = F.smooth_l1_loss

    def forward(self, x, *args):
        #print('original',x)
        x = self.instancenorm(x)
        #print('normalized',x)
        x = self.flatten(x)
        #print('flattened',x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = self.fc5(x)
        x = self.tanh(x)
        x = self.fc6(x)
        x = self.tanh(x)
        x = self.fc7(x)
        return x

'''
TEST network
'''
class build_network_TEST(nn.Module):
    def __init__(self, input_size, output_size):
        super(build_network_TEST, self).__init__()
        # size of layers
        layer_1 = 64
        layer_2 = 64
        layer_3 = 48
        layer_4 = 32
        layer_5 = 16
        layer_6 = 16
        # FCNN
        self.fc1 = nn.Linear(input_size, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, layer_3)
        self.fc4 = nn.Linear(layer_3, layer_4)
        self.fc5 = nn.Linear(layer_4, layer_5)
        self.fc6 = nn.Linear(layer_5, layer_6)
        self.fc7 = nn.Linear(layer_6, output_size)
        # activation functions
        self.tanh = nn.Tanh()
        self.instancenorm = nn.InstanceNorm1d(input_size)
        self.flatten = nn.Flatten()
        # Huber loss function
        self.loss_func = F.smooth_l1_loss

    def forward(self, x, *args):
        #print('original',x)
        x = self.instancenorm(x)
        #print('normalized',x)
        x = self.flatten(x)
        #print('flattened',x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = self.fc5(x)
        x = self.tanh(x)
        x = self.fc6(x)
        x = self.tanh(x)
        x = self.fc7(x)
        return x


