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
import newSimulatorBasedOnJava.rule.operation.basic.SPT as SPT
import newSimulatorBasedOnJava.rule.operation.basic.FCFS as FCFS
import newSimulatorBasedOnJava.rule.operation.basic.WINQ as WINQ
import newSimulatorBasedOnJava.rule.operation.basic.MS as MS
import newSimulatorBasedOnJava.rule.operation.basic.CR as CR
import newSimulatorBasedOnJava.simulation.SequencingDecisionSituation as SequencingDecisionSituation
import newSimulatorBasedOnJava.jobshop.LastOperationDownloadProcess as LastOperationDownloadProcess
import newSimulatorBasedOnJava.jobshop.Process as Process
import newSimulatorBasedOnJava.jobshop.TransportationProcess as TransportationProcess

import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadFinishEvent as FirstOperationUploadFinishEvent
import newSimulatorBasedOnJava.simulation.event.FirstOperationUploadStartEvent as FirstOperationUploadStartEvent
import newSimulatorBasedOnJava.simulation.event.JobArrivalEvent as JobArrivalEvent
import newSimulatorBasedOnJava.simulation.event.LastOperationDownloadStartEvent as LastOperationDownloadStartEvent
import newSimulatorBasedOnJava.simulation.event.OperationVisitEvent as OperationVisitEvent
import newSimulatorBasedOnJava.simulation.event.ProcessStartEvent as ProcessStartEvent
import newSimulatorBasedOnJava.simulation.event.TransportationStartEvent as TransportationStartEvent
import newSimulatorBasedOnJava.rule.RuleType as RuleType
import newSimulatorBasedOnJava.RL.sequencingRuleTrain.SimulationForSequencingRuleTrain as SimulationForSequencingRuleTrain


class SimulationForRLsequencingRuleTest(SimulationForSequencingRuleTrain.SimulationForSequencingRuleTrain):
    def __init__(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, minNumOperations, maxNumOperations,
                utilLevel, dueDateFactor, revisit, **kwargs):
        SimulationForSequencingRuleTrain.SimulationForSequencingRuleTrain.__init__(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, minNumOperations, maxNumOperations,
                utilLevel, dueDateFactor, revisit, sequeningTest = True, routingTest = False, run_seed = kwargs['run_seed'], objective = kwargs['objective'], MC = 1,  reward_function = 1, IQL = 0, I_DDQN = 0)

        # the following is used for RL
        # each item is a network
        self.net_dict = {}
        self.func_list = [SPT.SPT(RuleType.RuleType.SEQUENCING), WINQ.WINQ(RuleType.RuleType.SEQUENCING),
                          MS.MS(RuleType.RuleType.SEQUENCING), CR.CR(RuleType.RuleType.SEQUENCING)]
        self.output_size = len(self.func_list)
        self.address_seed = "{}/sequencing_models/scenario_" + self.dataset_name + "/run_" + str(seed) + "_MC_rwd" + str(
            kwargs['reward_function']) + ".pt"  # modified by mengxu
        self.input_size = 25 # related to sequencing_data
        self.network = network_validated(self.input_size, self.output_size)
        self.network.network.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
        self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
        self.build_state = self.state_multi_channel
        self.bad_run_check = True

        if 'bad_run_check' in kwargs:
            self.bad_run_check = kwargs['bad_run_check']

        self.setup()


    def run(self):
        self.count = 0
        while not self.eventQueue.empty() and self.throughput < self.numJobsRecorded:
            nextEvent = self.eventQueue.pop()
            self.beforeThroughput = self.throughput
            self.systemState.setClockTime(nextEvent.getTime())
            if isinstance(nextEvent, ProcessFinishEvent.ProcessFinishEvent):
                self.before_operation(nextEvent)
                worked = self.triggerProcessFinishEvent(nextEvent)
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
                    print("Number warm-up job completed: " + str(len(self.systemState.warmupJobsCompleted)))
                    print("Number job completed: " + str(len(self.systemState.getJobsCompleted())))
                    self.eventQueue.clear()
                for workCenter in self.systemState.getWorkCenters():
                    numOpsInQueue = workCenter.numOpsInQueue()
                    if numOpsInQueue > 100:
                        print("bad run: numOpsInQueue > 100")
                        print("Number warm-up job completed: " + str(len(self.systemState.warmupJobsCompleted)))
                        print("Number job completed: " + str(len(self.systemState.getJobsCompleted())))
                        self.systemState.setClockTime(np.Infinity)
                        self.eventQueue.clear()


    def triggerProcessFinishEvent(self, processFinish):
        workCenter = processFinish.process.getWorkCenter()
        processFinish.process.getOperationOption().getJob().addProcessFinishEvent(self)
        if len(workCenter.getQueue()) > 1:
            sequencingDecisionSituation = SequencingDecisionSituation.SequencingDecisionSituation(workCenter.getQueue(), workCenter, self.getSystemState())
            # hand over the target machines' sequencing function to policy network
            idx_seq, s_t = self.action_sqc_rule(self.sequencing_data_generation(processFinish), processFinish)
            selectedSequencingRule = self.func_list[idx_seq]
            self.setSequencingRule(selectedSequencingRule)
            dispatchedOp, op_idx = self.getSequencingRule().priorOperation(sequencingDecisionSituation)
            workCenter.removeFromQueue(dispatchedOp)
            processStartTime = np.maximum(workCenter.getReadyTime(), processFinish.time)
            nextP = Process.Process(workCenter, processFinish.process.getMachineId(), dispatchedOp, processStartTime)
            self.addEvent(ProcessStartEvent.ProcessStartEvent(nextP))
        elif len(workCenter.getQueue()) == 1:
            dispatchedOp = workCenter.getQueue()[0]
            workCenter.removeFromQueue(dispatchedOp)
            processStartTime = np.maximum(workCenter.getReadyTime(), processFinish.time)
            nextP = Process.Process(workCenter, processFinish.process.getMachineId(), dispatchedOp, processStartTime)
            self.addEvent(ProcessStartEvent.ProcessStartEvent(nextP))
            worked = True

        nextOp = processFinish.process.getOperationOption().getNext(self.getSystemState(), self.getRoutingRule())
        if nextOp == None:
            workCenterID = processFinish.process.getOperationOption().getWorkCenter().getId()
            downloadTime = self.getWorkCenterExitTranserTime(workCenterID)
            processFinish.process.getOperationOption().getOperation().setDownloadTimeToSelectedWorkCenter(
                downloadTime)
            downloadProcess = LastOperationDownloadProcess.LastOperationDownloadProcess(
                processFinish.process.getWorkCenter(), processFinish.process.getOperationOption(),
                processFinish.time)
            self.addEvent(LastOperationDownloadStartEvent.LastOperationDownloadStartEvent(downloadProcess))
        else:
            nextOp.getOperation().setSelectedWorkCenter(nextOp.getWorkCenter())
            fromWorkCenterID = processFinish.process.getOperationOption().getWorkCenter().getId()
            toWorkCenterID = nextOp.getWorkCenter().getId()
            transferTime = self.getWorkCenterTranserTime(fromWorkCenterID, toWorkCenterID)
            nextOp.getOperation().setTransferTimeToSelectedWorkCenter(transferTime)
            transportationProcess = TransportationProcess.TransportationProcess(
                processFinish.process.getOperationOption().getWorkCenter(), nextOp.getWorkCenter(), nextOp,
                processFinish.time)
            self.addEvent(TransportationStartEvent.TransportationStartEvent(transportationProcess))
        return True

    def action_sqc_rule(self, local_data, processFinish):
        s_t = self.build_state(local_data, processFinish)
        value = self.network.forward(s_t.reshape([1, 1, self.input_size]))
        a_t = torch.argmax(value)
        return a_t,s_t


'''
classes of ANN (along with optimzers), total 3 types:
1. normalization (normalize data of same type in one channel)
2. abstract (use the reciprocal of input data, no normalization layers)
3. multi-channel (normalize data in multiple channels)
'''

class network_validated(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_validated, self).__init__()
        self.lr = 0.001
        self.input_size = input_size
        self.output_size = output_size
        # for slicing the data
        self.no_size = 3
        self.pt_size = 6
        self.remaining_pt_size = 11
        self.ttd_slack_size = 16
        # FCNN parameters
        layer_1 = 48
        layer_2 = 36
        layer_3 = 36
        layer_4 = 24
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.normlayer_no = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_pt = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_remaining_pt = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        self.normlayer_ttd_slack = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.subsequent_module = nn.Sequential(
                                nn.Linear(self.input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the universal network for all scheudling agents
        self.network = nn.ModuleList([self.normlayer_no, self.normlayer_pt, self.normlayer_remaining_pt, self.normlayer_ttd_slack, self.subsequent_module])
        # accompanied by optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, *args):
        #print('original',x)
        # slice the data
        x_no = x[:,:, : self.no_size]
        x_pt = x[:,:, self.no_size : self.pt_size]
        x_remaining_pt = x[:,:, self.pt_size : self.remaining_pt_size]
        x_ttd_slack = x[:,:, self.remaining_pt_size : self.ttd_slack_size]
        x_rest = x[:,:, self.ttd_slack_size :].squeeze(1)
        # normalize data in multiple channels
        x_normed_no = self.network[0](x_no)
        x_normed_pt = self.network[1](x_pt)
        x_normed_remaining_pt = self.network[2](x_remaining_pt)
        x_normed_ttd_slack = self.network[3](x_ttd_slack)
        #print('normalized',x_normed_no)
        # concatenate all data
        #print(x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd, x_normed_slack, x_rest)
        x = torch.cat([x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd_slack, x_rest], dim=1)
        #print('combined',x)
        # the last, independent part of module
        x = self.network[4](x)
        #print('output',x)
        return x

class network_value_based(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_value_based, self).__init__()
        self.lr = 0.005
        self.input_size = input_size
        self.output_size = output_size
        self.flattened_input_size = torch.tensor(self.input_size).prod()
        # FCNN parameters
        layer_1 = 64
        layer_2 = 48
        layer_3 = 48
        layer_4 = 36
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.norm_layer = nn.Sequential(
                                nn.LayerNorm(self.input_size),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.FC_layers = nn.Sequential(
                                nn.Linear(self.flattened_input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the universal network for all scheudling agents
        self.network = nn.ModuleList([self.norm_layer, self.FC_layers])
        # accompanied by optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, *args):
        #print('original',x)
        x = self.network[0](x)
        x = self.network[1](x)
        #print('output',x)
        return x
