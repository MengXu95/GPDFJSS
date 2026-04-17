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


class SimulationForRoutingRuleTrain(HeterogeneousSimulation.HeterogeneousSimulation):
    def __init__(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, minNumOperations, maxNumOperations,
                utilLevel, dueDateFactor, revisit, **kwargs):
        HeterogeneousSimulation.HeterogeneousSimulation.__init__(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, minNumOperations, maxNumOperations,
                utilLevel, dueDateFactor, revisit, routingTrain = True)

        # the following is used for RL
        self.run_seed = kwargs['run_seed']
        np.random.seed(self.run_seed)
        self.incomplete_rep_memo_all = {}
        # the reward record
        self.sqc_reward_record_job = []
        self.rt_reward_record_job = []
        self.routing_data = None
        self.initialised = False
        self.afterRandomExploration = False
        # self.hash_set=set()
        # standby disctionary for storing incomplete experience for learning
        self.build_routing_experience = self.complete_experience_full
        self.incomplete_experience = {}
        # list that contains available rules, and use SPT for the first phase of warmup
        self.func_list = [LRT.LRT(RuleType.RuleType.ROUTING), SBT.SBT(RuleType.RuleType.ROUTING),
                          SRT.SRT(RuleType.RuleType.ROUTING), WIQ.WIQ(RuleType.RuleType.ROUTING)]
        # # list that contains available rules, and use SPT for the first phase of warmup
        # self.func_list = [SPT.SPT(RuleType.RuleType.SEQUENCING), WINQ.WINQ(RuleType.RuleType.SEQUENCING), MS.MS(RuleType.RuleType.SEQUENCING), CR.CR(RuleType.RuleType.SEQUENCING)]
        self.func_selection = 0
        # state space, eah machine generate 3 types of data, along with the processing time of arriving job, and job slack
        # self.input_size = len(self.func_list) * 3 + 3
        self.input_size = self.numWorkCenters * 3 + 3
        # action space, consists of all selectable rules
        self.output_size = len(self.func_list)
        # learning rate
        self.lr = 0.01
        # specify the address to store the model
        print("---> DEFAULT mode ON <---")
        self.dataset_name = kwargs['objective'] + "_" + str(utilLevel) + "_" + str(dueDateFactor)
        if len(self.func_list) == 2:
            self.routing_action_NN = build_network_small(self.input_size, self.output_size)
            # self.address_seed = "{}/routing_models/small_state_dict" + '{}wc{}m'.format(len(wc_list), len(m_list)) #original
            self.address_seed = "{}/routing_models/scenario_" + self.dataset_name + "/run_" + str(seed) + "_small_state_dict"+'{}wc{}m'.format(self.input_size,self.output_size) #modified by mengxu 2022.10.31
        elif len(self.func_list) == 3:
            self.routing_action_NN = build_network_medium(self.input_size, self.output_size)
            self.address_seed = "{}/routing_models/scenario_" + self.dataset_name + "/run_" + str(seed) + "_medium_state_dict"+'{}wc{}m'.format(self.input_size,self.output_size) #modified by mengxu 2022.10.31
        elif len(self.func_list) == 4:
            self.routing_action_NN = build_network_large(self.input_size, self.output_size)
            self.address_seed = "{}/routing_models/scenario_" + self.dataset_name + "/run_" + str(seed) + "_large_state_dict"+'{}wc{}m'.format(self.input_size,self.output_size) #modified by mengxu 2022.10.31
        # shared functions
        self.routing_target_NN = copy.deepcopy(self.routing_action_NN)
        self.build_state = self.state_deeper
        self.train = self.train_DDQN

        ''' specify the optimizer '''
        self.optimizer = optim.SGD(self.routing_action_NN.parameters(), lr=self.lr, momentum = 0.9)
        # Initialize the parameters for training
        self.minibatch_size = 128
        self.rep_memo_size = 1024
        self.discount_factor = 0.8 # how much agent care long-term rewards
        self.epsilon = 0.5  # exploration
        # parameters of training
        self.routing_action_NN_training_interval = 2
        self.routing_action_NN_training_time_record = []
        self.routing_target_NN_sync_interval = 250  # synchronize the weights of NN every 500 time units
        self.routing_target_NN_update_time_record = []
        # initialize the list for deta memory
        self.rep_memo_all = []
        self.data_memory = []
        self.exploration_record = []
        self.time_record = []
        self.loss_record = []
        # processes
        # self.setup() # original for training
        if 'sequencingTest' in kwargs and not kwargs['sequencingTest']:
            self.setup()

        if 'routingTest' in kwargs and not kwargs['routingTest']:
            self.setup()

    def getEnvTime(self):
        return self.throughput
        # return self.numJobsArrived # original used one


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
                    self.systemState.setClockTime(np.Infinity)
                    self.eventQueue.clear()
                    print("bad run: numOpsInQueue > 100")
        # end the training after span time
        # and store the trained parameters
        print('Final replay_memory is:\n', 'size:', len(self.rep_memo_all), tabulate(self.rep_memo_all, headers=['s_t', 'a_t', 'r_t', 's_t+1']))
        # save the parameters of policy / action network after training
        torch.save(self.routing_action_NN.state_dict(), self.address_seed.format(sys.path[0]))
        print("Training terminated, store trained parameters to: {}".format(self.address_seed))
        self.saveLoss()

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
            '''
            Phase 1.1 : warm-up
            within this phase, agent shift between sequencing rules
            '''
            # take over the target machines' sequencing function
            # from FIFO to here
            # first half of warm-up period
            period = self.warmupJobs / 5
            # print("num job arrived: " + str(self.numJobsArrived))
            if self.numJobsArrived < period * len(self.func_list):
                idx_rout = int(self.numJobsArrived / period)  # todo: need to double check this function
                self.func_selection = idx_rout
                selectedRoutingRule = self.func_list[idx_rout]
                self.setRoutingRule(selectedRoutingRule)
                job_pt = [-1 for i in range(self.numWorkCenters)] # -1 represents this workCenter can not process this operation by mengxu 2022.11.25
                # we use the above function (considering the workcenter do not have processing ability for this operation) is because we need to make sure the input size is always same
                for opeOption in nextOperation.getOperationOptions():
                    wc_id = opeOption.getWorkCenter().getId()
                    job_pt[wc_id] = opeOption.getProcTime()
                job_pt = np.array(job_pt)
                job_slack = nextOperation.getJob().getSlack(self.systemState.getClockTime())
                # this is the estimated slack time and I have not considered the transfer time by mengxu 2022.11.25
                routing_data = self.routing_data_generation(nextOperation)
                s_t = self.build_state(routing_data, job_pt, job_slack, nextOperation)
                a_t = torch.tensor(self.func_selection)
                job_idx = nextOperation.getJob().getId()
                operationOption = processFinish.process.getOperationOption().getNext(self.getSystemState(),self.getRoutingRule())
                self.build_experience(job_idx, s_t, a_t,0)  # I think this should be the whole simulation, so should be 0 or just delete by mengxu 2022.11.23
                self.time_record.append(self.numJobsArrived)
                operationOption.getOperation().setSelectedWorkCenter(operationOption.getWorkCenter())
                fromWorkCenterID = processFinish.process.getOperationOption().getWorkCenter().getId()
                toWorkCenterID = operationOption.getWorkCenter().getId()
                transferTime = self.getWorkCenterTranserTime(fromWorkCenterID, toWorkCenterID)
                operationOption.getOperation().setTransferTimeToSelectedWorkCenter(transferTime)
                transportationProcess = TransportationProcess.TransportationProcess(
                    processFinish.process.getOperationOption().getWorkCenter(), operationOption.getWorkCenter(), operationOption,
                    processFinish.time)
                self.addEvent(TransportationStartEvent.TransportationStartEvent(transportationProcess))
                # calculate slack gain/loss
                current_slack = nextOperation.getJob().getSlack(self.systemState.getClockTime())
                estimated_slack_time = estimated_job_slack
                slack_change = current_slack - estimated_slack_time
                critical_level_R = 1 - estimated_slack_time / 100
                self.build_routing_experience(job_idx, routing_data, job_pt, slack_change, critical_level_R, nextOperation, 0)
            '''
            Phase 1.2 : random exploration
            within this phase, agents choose random sequencing action, to accumulate experience
            '''
            if self.numJobsArrived < self.warmupJobs and self.numJobsArrived >= period * len(self.func_list):
                # change the target machines' sequencing function to random exploration, to try all combinations
                # second half of warm-up period
                if self.afterRandomExploration == False:
                    # print("num job arrived: " + str(self.numJobsArrived))
                    print("start random exploration before the arrived job number is smaller than {}".format(
                        self.warmupJobs))
                    self.afterRandomExploration = True
                job_idx = nextOperation.getJob().getId()
                job_pt = [-1 for i in range(self.numWorkCenters)]
                for opeOption in nextOperation.getOperationOptions():
                    wc_id = opeOption.getWorkCenter().getId()
                    job_pt[wc_id] = opeOption.getProcTime()
                    # job_pt.append(opeOption.getProcTime())
                job_pt = np.array(job_pt)
                job_slack = nextOperation.getJob().getSlack(self.systemState.getClockTime())
                routing_data = self.routing_data_generation(nextOperation)
                idx_rout = self.action_random_exploration(job_idx, routing_data, job_pt, job_slack, 0, nextOperation)
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
                # calculate slack gain/loss
                current_slack = nextOperation.getJob().getSlack(self.systemState.getClockTime())
                estimated_slack_time = estimated_job_slack
                slack_change = current_slack - estimated_slack_time
                critical_level_R = 1 - estimated_slack_time / 100
                self.build_routing_experience(job_idx, routing_data, job_pt, slack_change, critical_level_R, nextOperation, 0)
            if self.getEnvTime() < self.numJobsRecorded and self.numJobsArrived >= self.warmupJobs and self.initialised == False:
                '''
                After the warm up, build initial replay memory
                and hand over to sequencing NN
                '''
                # after the warm up period, build replay memory and start training
                # self.build_initial_rep_memo()
                self.build_initial_rep_memo_parameter_sharing()
                # from traditional def training_process_parameter_sharing(self):
                # pre-train the policy NN before hand over to it
                for i in range(20):
                    self.train()
                self.reduction = (self.lr - self.lr / 10) / 10
                self.initialised = True
            if self.getEnvTime() < self.numJobsRecorded and self.numJobsArrived >= self.warmupJobs:
                if self.numJobsArrived % self.routing_action_NN_training_interval == 0:
                    # periodic training
                    self.train()
                if int(self.numJobsArrived) % int((self.numJobsRecorded - self.warmupJobs) / 10) == 0:
                    # reduce the learning rate
                    if self.lr - self.reduction > 0: # modified by mengxu 2022.11.23
                        self.lr -= self.reduction
                    self.optimizer = optim.SGD(self.routing_action_NN.parameters(), lr=self.lr, momentum=0.9)
                    self.epsilon -= 0.01
                    print('--------------------------------------------------------')
                    print('learning rate adjusted to {} at time {}'.format(self.lr, self.numJobsArrived))
                    print('--------------------------------------------------------')
                # hand over the target machines' sequencing function to policy network
                job_idx = nextOperation.getJob().getId()
                job_pt = [-1 for i in range(self.numWorkCenters)]
                for opeOption in nextOperation.getOperationOptions():
                    wc_id = opeOption.getWorkCenter().getId()
                    job_pt[wc_id] = opeOption.getProcTime()
                    # job_pt.append(opeOption.getProcTime())
                job_pt = np.array(job_pt)
                job_slack = nextOperation.getJob().getSlack(self.systemState.getClockTime())
                routing_data = self.routing_data_generation(nextOperation)
                idx_rout = self.action_DRL(job_idx, routing_data, job_pt, job_slack, 0, nextOperation)
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
                # calculate slack gain/loss
                current_slack = nextOperation.getJob().getSlack(self.systemState.getClockTime())
                estimated_slack_time = estimated_job_slack
                slack_change = current_slack - estimated_slack_time
                critical_level_R = 1 - estimated_slack_time / 100
                self.build_routing_experience(job_idx, routing_data, job_pt, slack_change, critical_level_R, nextOperation, 0)
                # update the replay memory periodically
                if self.numJobsArrived % (self.routing_action_NN_training_interval * 10) == 0:
                    if len(self.rep_memo_all) > self.rep_memo_size:
                        truncation = len(self.rep_memo_all) - self.rep_memo_size
                        self.rep_memo_all = self.rep_memo_all[truncation:]
                # synchronize the parameter of policy and target network
                if self.numJobsArrived % self.routing_target_NN_sync_interval == 0:
                    self.routing_target_NN = copy.deepcopy(self.routing_action_NN)
                    print('--------------------------------------------------------')
                    print('the target network and epsilion are updated at time %s' % self.numJobsArrived)
                    print('--------------------------------------------------------')



    def triggerJobArrivalEvent(self, jobArrival):
        operation = jobArrival.job.getOperation(0)
        estimated_job_slack = operation.getJob().getSlack(self.systemState.getClockTime())
        for op in operation.getOperationOptions():
            op.setReadyTime(jobArrival.job.getReleaseTime())
        routingDecisionSituation = RoutingDecisionSituation.RoutingDecisionSituation(operation.getOperationOptions(), self.getSystemState())

        '''
        Phase 1.1 : warm-up
        within this phase, agent shift between sequencing rules
        '''
        # take over the target machines' sequencing function
        # from FIFO to here
        # first half of warm-up period
        period = self.warmupJobs/5
        # print("num job arrived: " + str(self.numJobsArrived))
        if self.numJobsArrived < period * len(self.func_list):
            idx_rout = int(self.numJobsArrived/period) #todo: need to double check this function
            self.func_selection = idx_rout
            selectedRoutingRule = self.func_list[idx_rout]
            self.setRoutingRule(selectedRoutingRule)
            job_pt = [-1 for i in range(self.numWorkCenters)]
            for opeOption in operation.getOperationOptions():
                wc_id = opeOption.getWorkCenter().getId()
                job_pt[wc_id] = opeOption.getProcTime()
                # job_pt.append(opeOption.getProcTime())
            job_pt = np.array(job_pt)
            job_slack = jobArrival.job.getSlack(self.systemState.getClockTime())
            routing_data = self.routing_data_generation(operation)
            s_t = self.build_state(routing_data, job_pt, job_slack, operation)
            a_t = torch.tensor(self.func_selection)
            job_idx = operation.getJob().getId()
            operationOption= self.getRoutingRule().nextOperationOption(routingDecisionSituation)
            selectedWorkCenterID = operationOption.getWorkCenter().getId()
            self.build_experience(job_idx, s_t, a_t, 0) # I think this should be the whole simulation, so should be 0 or just delete by mengxu 2022.11.23
            self.time_record.append(self.numJobsArrived)
            uploadTime = self.getEntryWorkCenterTranserTime(selectedWorkCenterID)
            operationOption.getOperation().setSelectedWorkCenter(operationOption.getWorkCenter())
            operationOption.getOperation().setUploadTimeToSelectedWorkCenter(uploadTime)
            process = FirstOperationUploadProcess.FirstOperationUploadProcess(operationOption.getWorkCenter(),
                                                                              operationOption,
                                                                              jobArrival.job.getReleaseTime())
            self.addEvent(FirstOperationUploadStartEvent.FirstOperationUploadStartEvent(process))
            # calculate slack gain/loss
            current_slack = jobArrival.job.getSlack(self.systemState.getClockTime())
            estimated_slack_time = estimated_job_slack
            slack_change = current_slack - estimated_slack_time
            critical_level_R = 1 - estimated_slack_time / 100
            self.build_routing_experience(job_idx, routing_data, job_pt, slack_change, critical_level_R, operation, 0)
        '''
        Phase 1.2 : random exploration
        within this phase, agents choose random sequencing action, to accumulate experience
        '''
        if self.numJobsArrived < self.warmupJobs and self.numJobsArrived >= period*len(self.func_list):
            # change the target machines' sequencing function to random exploration, to try all combinations
            # second half of warm-up period
            if self.afterRandomExploration == False:
                # print("num job arrived: " + str(self.numJobsArrived))
                print("start random exploration before the arrived job number is smaller than {}".format(
                    self.warmupJobs))
                self.afterRandomExploration = True
            job_idx = operation.getJob().getId()
            job_pt = [-1 for i in range(self.numWorkCenters)]
            for opeOption in operation.getOperationOptions():
                wc_id = opeOption.getWorkCenter().getId()
                job_pt[wc_id] = opeOption.getProcTime()
                # job_pt.append(opeOption.getProcTime())
            job_pt = np.array(job_pt)
            job_slack = jobArrival.job.getSlack(self.systemState.getClockTime())
            routing_data = self.routing_data_generation(operation)
            idx_rout = self.action_random_exploration(job_idx, routing_data, job_pt, job_slack, 0, operation)
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
            # calculate slack gain/loss
            current_slack = jobArrival.job.getSlack(self.systemState.getClockTime())
            estimated_slack_time = estimated_job_slack
            slack_change = current_slack - estimated_slack_time
            critical_level_R = 1 - estimated_slack_time / 100
            self.build_routing_experience(job_idx, routing_data, job_pt, slack_change, critical_level_R, operation, 0)

        if self.getEnvTime() < self.numJobsRecorded and self.numJobsArrived >= self.warmupJobs and self.initialised == False:
            '''
            After the warm up, build initial replay memory
            and hand over to sequencing NN
            '''
            # after the warm up period, build replay memory and start training
            # self.build_initial_rep_memo()
            self.build_initial_rep_memo_parameter_sharing()
            # from traditional def training_process_parameter_sharing(self):
            # pre-train the policy NN before hand over to it
            for i in range(20):
                self.train()
            self.reduction = (self.lr - self.lr/10)/10
            self.initialised = True
        if self.getEnvTime() < self.numJobsRecorded and self.numJobsArrived >= self.warmupJobs:
            if self.numJobsArrived%self.routing_action_NN_training_interval == 0:
                # periodic training
                self.train()
            if int(self.numJobsArrived) % int((self.numJobsRecorded-self.warmupJobs)/10) == 0:
                # reduce the learning rate
                if self.lr - self.reduction > 0:  # modified by mengxu 2022.11.23
                    self.lr -= self.reduction
                self.optimizer = optim.SGD(self.routing_action_NN.parameters(), lr=self.lr, momentum=0.9)
                self.epsilon -= 0.01
                print('--------------------------------------------------------')
                print('learning rate adjusted to {} at time {}'.format(self.lr, self.numJobsArrived))
                print('--------------------------------------------------------')
            # hand over the target machines' sequencing function to policy network
            job_idx = operation.getJob().getId()
            job_pt = [-1 for i in range(self.numWorkCenters)]
            for opeOption in operation.getOperationOptions():
                wc_id = opeOption.getWorkCenter().getId()
                job_pt[wc_id] = opeOption.getProcTime()
                # job_pt.append(opeOption.getProcTime())
            job_pt = np.array(job_pt)
            job_slack = jobArrival.job.getSlack(self.systemState.getClockTime())
            routing_data = self.routing_data_generation(operation)
            idx_rout= self.action_DRL(job_idx, routing_data, job_pt, job_slack, 0, operation)
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
            # calculate slack gain/loss
            current_slack = jobArrival.job.getSlack(self.systemState.getClockTime())
            estimated_slack_time = estimated_job_slack
            slack_change = current_slack - estimated_slack_time
            critical_level_R = 1 - estimated_slack_time / 100
            self.build_routing_experience(job_idx, routing_data, job_pt, slack_change, critical_level_R, operation, 0)
            # update the replay memory periodically
            if self.numJobsArrived % (self.routing_action_NN_training_interval*10) == 0:
                if len(self.rep_memo_all) > self.rep_memo_size:
                    truncation = len(self.rep_memo_all) - self.rep_memo_size
                    self.rep_memo_all = self.rep_memo_all[truncation:]
            # synchronize the parameter of policy and target network
            if self.numJobsArrived % self.routing_target_NN_sync_interval == 0:
                self.routing_target_NN = copy.deepcopy(self.routing_action_NN)
                print('--------------------------------------------------------')
                print('the target network and epsilion are updated at time %s' % self.numJobsArrived)
                print('--------------------------------------------------------')


    # give out the information related to routing decision
    def routing_data_generation(self, operation):
        # note that we subtract current time from available_time
        # todo: the data used for selecting a machine from the queue by routing rule
        # becasue state_update_all function may be called at a different time
        # modified by mengxu
        routing_data_all=[[-1, -1, -1, -1] for i in range(self.numWorkCenters)] # todo: should consider all the workcenter even it is not allowed to process the operation!!! 2022.11.23 by mengxu
        # todo: so if it is not allowed to process, the information of this workcenter is all about -1
        operationOptions = operation.getOperationOptions()
        for operationOption in operationOptions:
            workCenter = operationOption.getWorkCenter()
            wc_id = workCenter.getId()
            cumulative_pt = workCenter.getWorkInQueue()
            waiting_time = max(0, workCenter.getReadyTime() - self.systemState.getClockTime())
            que_size = workCenter.numOpsInQueue()
            cumulative_run_time = workCenter.getBusyTime()
            routing_data_wc = [cumulative_pt, waiting_time, que_size, cumulative_run_time]
            routing_data_all[wc_id] = routing_data_wc
            # routing_data_all.append(routing_data_wc)
        # self.routing_data = [self.cumulative_pt, max(0,self.available_time-self.env.now), self.que_size, self.cumulative_run_time] #original
        return routing_data_all

    # random exploration is for collecting more experience
    def action_random_exploration(self, job_idx, routing_data, job_pt, job_slack, wc_idx, operation, *args):
        s_t = self.build_state(routing_data, job_pt, job_slack, operation)
        # generate a random number in [0, self.m_per_wc)
        a_t = torch.randint(0, len(self.func_list), [])
        # add current state and action to target wc's incomplete experience
        self.build_experience(job_idx, s_t, a_t, wc_idx)
        self.time_record.append(self.numJobsArrived)
        print('RANDOM ROUTING: job {} assigned by routing rule {}'.format(job_idx,self.func_list[a_t].getName()))
        return a_t

    def action_DRL(self, job_idx, routing_data, job_pt, job_slack, wc_idx, operation, *args):
        s_t = self.build_state(routing_data, job_pt, job_slack, operation)
        # generate the action
        if random.random() < self.epsilon:
            a_t = torch.randint(0, len(self.func_list), [])
            # print('RANDOM ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx,self.wc_list[wc_idx].m_list[a_t].m_idx))
        else:
            # input state to policy network, produce the state-action value
            value = self.routing_action_NN.forward(s_t.reshape(1, 1, self.input_size), wc_idx)
            # print("State:",s_t, 'State-Action Values:', value)
            a_t = torch.argmax(value)
            # print('DRL ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx,self.wc_list[wc_idx].m_list[a_t].m_idx))
        # add current state and action to target wc's incomplete experience
        # this is the first half of a single record of experience
        # the second half will be appended to first half when routed job complete its operation
        self.build_experience(job_idx, s_t, a_t, wc_idx)
        self.time_record.append(self.numJobsArrived)
        return a_t

    def build_experience(self, job_idx, s_t, a_t, wc_idx):
        self.incomplete_experience[job_idx] = [s_t, a_t]

    '''
    2. downwards are functions used for building the state of the experience (replay memory)
    '''

    # def state_normalization(self, routing_data, job_pt, job_slack, wc_idx):
    #     coming_job_idx = np.where(self.job_creator.next_wc_list == wc_idx)[0]  # return the index of coming jobs
    #     coming_job_no = coming_job_idx.size  # expected arriving job number
    #     if coming_job_no:  # if there're jobs coming at your workcenter
    #         next_job = self.job_creator.release_time_list[coming_job_idx].argmin()  # the index of next job
    #         coming_job_time = (self.job_creator.release_time_list[coming_job_idx] - self.env.now)[
    #             next_job]  # time from now when next job arrives at workcenter
    #         coming_job_slack = self.job_creator.arriving_job_slack_list[coming_job_idx][
    #             next_job]  # what's the average slack time of the arriving job
    #     else:
    #         coming_job_time = 0
    #         coming_job_slack = 0
    #     m_state = [a[:2] for a in routing_data]
    #     s_t = torch.tensor(np.concatenate(m_state + [job_pt, [job_slack, coming_job_time, coming_job_slack]]),
    #                        dtype=torch.float)
    #     return s_t

    def state_deeper(self, routing_data, job_pt, job_slack, operation):
        coming_job_idx = operation.getJob().getId()  # return the index of coming jobs
        coming_job_no = 1  # expected arriving job number
        if coming_job_no:  # if there're jobs coming at your workcenter
            # modified 2022.11.25 time from now when next job arrives at workcenter I think this should be the transfer time, but we do not know the transfer time, so this is just the current time or 0?
            # todo: need to double check!!! 2022.11.25
            coming_job_time = 0  # coming_job_time = operation.getJob().getReleaseTime() - self.systemState.getClockTime()  # original time from now when next job arrives at workcenter
            coming_job_slack = operation.getJob().getSlack(self.systemState.getClockTime())  # what's the average slack time of the arriving job
        else:
            coming_job_time = 0
            coming_job_slack = 0
        m_state = [a[:2] for a in routing_data]
        data_job_related = [job_pt, [job_slack, coming_job_time, coming_job_slack]]
        s_t = torch.tensor(np.concatenate(m_state + data_job_related),
                           dtype=torch.float)
        return s_t

    def state_Lang2020(self, routing_data, job_pt, job_slack, wc_idx):
        m_state = [a[:1] for a in routing_data]
        s_t = torch.tensor(np.concatenate(m_state + [job_pt, [job_slack]]), dtype=torch.float)
        return s_t

    '''
    3. downwards are functions used for building / updating replay memory
    '''
    # called after the warm-up period
    def build_initial_rep_memo_parameter_sharing(self):
        print('INITIALIZATION - replay_memory is:\n', len(self.rep_memo_all), tabulate(self.rep_memo_all, headers=['s_t', 'a_t', 'r_t', 's_t+1']))
        print('input-size:', self.input_size, '\nreplay_memory_size:', len(self.rep_memo_all))
        print('---------------------------initialization accomplished-----------------------------')

    '''
    4. downwards are functions used in the training of DRL, including the dynamic training process
       dynamic training parameters update and the optimization funciton of ANN
       the class for builidng the ANN is at the bottom
    '''


    # the function that draws minibatch and trains the action NN
    def train_DDQN(self):
        print(".............TRAINING .............%s" % (self.numJobsArrived))
        """
        draw the random minibatch to train the network, randomly
        every element in the replay menory is a list [s_0, a_0, r_0, s_1]
        all element of this list are tensors
        """
        size = min(len(self.rep_memo_all), self.minibatch_size)
        minibatch = random.sample(self.rep_memo_all, size)
        '''
        slice, and stack the 1D tensors to several 2D tensors (batches)
        the "torch.stack" is only applicable when the augment is a list of tensors, or multi-dimensional tensor
        '''
        # reshape so we can meet the requiremnt of input of instancenorm
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape(size, 1, self.input_size)
        sample_s1_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size, 1, self.input_size)
        # reshape so we can use .gather() method
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size, 1)
        sample_r0_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape(size, 1)
        '''
        the size of these batches:
        sample_s0_batch = sample_s1_batch = minibatch size * 1 * input_size
        sample_a0_batch = sample_r0_batch = minibatch size * m_no
        sample_r0_batch = minibatch size
        '''
        # get the Q value (current value of state-action pair) of s0
        Q_0 = self.routing_action_NN.forward(sample_s0_batch)
        # print('Q_0 is:\n', Q_0)
        # print('a_0 is:', sample_a0_batch)
        # get the current state-action value of actions that would have been taken
        current_value = Q_0.gather(1, sample_a0_batch)
        # print('current value is:', current_value)
        '''
        get the Q Value of s_1 in both action and target network, to estimate the state value
        architecture is DDQN, NOT DQN !!!
        evaluate the greedy policy according to action network, but using the target network to estimate the value
        '''
        Q_1_action = self.routing_action_NN.forward(sample_s1_batch)
        Q_1_target = self.routing_target_NN.forward(sample_s1_batch)
        # print('Q_1_action is:\n', Q_1_action)
        # print('Q_1_target is:\n', Q_1_target)
        '''
        size of Q_0, Q_1_action and Q_1_target = minibatch size * m_no
        they're 2D tensors
        '''
        max_Q_1_action, max_Q_1_action_idx = torch.max(Q_1_action,
                                                       dim=1)  # use action network to get action, rather than max operation
        # print('max value of Q_1_action is:\n', max_Q_1_action)
        max_Q_1_action_idx = max_Q_1_action_idx.reshape([size, 1])
        # print('max idx of Q_1_action is:\n', max_Q_1_action_idx)
        # adjust the max_Q of s_0 by the discount factor (refer to Bellman Equation and TD)
        next_state_value = Q_1_target.gather(1, max_Q_1_action_idx)
        # print('estimated value of next state is:\n', next_state_value)
        next_state_value *= self.discount_factor
        # print('discounted next state value is:\n', next_state_value)
        '''
        the sum of reward and discounted max_Q is the target value
        target value is 2D matrix, size = minibatch_size * m_no
        '''
        # print('reward batch is:', sample_r0_batch)
        target_value = (sample_r0_batch + next_state_value)
        # print('target value is:', target_value)
        '''
        the loss is difference between current state-action value and the target value
        '''
        # calculate the loss
        loss = self.routing_action_NN.loss_func(current_value, target_value)
        print('loss is:', loss)
        self.loss_record.append(float(loss))
        # clear the gradient
        self.optimizer.zero_grad()  # zero the gradient buffers
        # calculate gradient
        loss.backward(retain_graph=True)
        # set the optimizer
        # print('perform the optimization of parameters')
        # optimize the parameters
        self.optimizer.step()

    def saveLoss(self):
        fileName = './routing_models/scenario_' + str(self.dataset_name) + '/' + str(self.run_seed) + '_training_loss'
        np.save(fileName, self.loss_record)

    # the function of building incomplete experience is in brain module
    # this function is called only if the routing_learning_event of constituent machine is triggered
    # when this function is called upon the completion of an operation
    def complete_experience_full(self, job_idx, routing_data, job_pt, slack_change, critical_level_R, operation, wc_idx):
        # first update current state, extract data from constituent machines
        # self.state_update_before_routing()
        # get the state at the time of job output
        s_t = self.build_state(routing_data, job_pt, 0, operation)
        # calculate reward of action
        # if slack time of job reduces, reward in [-1,0], if slack time increases, in [0,1]
        # print("slack_change: " + str(slack_change))
        r_t = torch.tensor(np.clip(slack_change * critical_level_R / 20, -1, 1), dtype=torch.float)
        self.rt_reward_record_job.append([self.numJobsArrived, r_t])
        # r_t = torch.tensor(np.clip(slack_change/40, -1, 1),dtype=torch.float)
        # print('build complete memory',job_idx,[r_t, s_t],self.env.now)
        # append reward and new state to corresponding incomplete experience
        self.incomplete_experience[job_idx] += [r_t, s_t]
        # and pop out this complete experience, add it to complete replay memory
        self.rep_memo_all.append(self.incomplete_experience.pop(job_idx))

    # def complete_experience_global_reward(self, job_idx, slack_change, critical_level_R):
    #     self.state_update_before_routing()
    #     s_t = self.build_state(self.routing_data, self.dummy_pt, 0, self.wc_idx)
    #     self.incomplete_experience_wc[job_idx] += [s_t]

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


