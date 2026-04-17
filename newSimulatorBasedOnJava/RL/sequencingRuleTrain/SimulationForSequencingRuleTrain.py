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


class SimulationForSequencingRuleTrain(HeterogeneousSimulation.HeterogeneousSimulation):
    def __init__(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, minNumOperations, maxNumOperations,
                utilLevel, dueDateFactor, revisit, **kwargs):
        HeterogeneousSimulation.HeterogeneousSimulation.__init__(self, seed, sequencingRule, routingRule, numWorkCenters, numJobsRecorded, warmupJobs, minNumOperations, maxNumOperations,
                utilLevel, dueDateFactor, revisit, sequencingTrain = True)

        # the following is used for RL
        self.run_seed = kwargs['run_seed']
        np.random.seed(self.run_seed)
        self.incomplete_rep_memo_machine = {}
        self.incomplete_rep_memo_job = {}
        # the reward record
        self.sqc_reward_record_job = []
        self.rt_reward_record_job = []
        # self.rep_memo_machine = {}
        self.rep_memo_job = {}
        self.sequencing_data = None
        self.curEvent = None
        self.curProcess = None
        self.initialised = False
        self.afterRandomExploration = False
        self.build_sqc_experience_repository_job()
        # list that contains available rules, and use SPT for the first phase of warmup
        self.func_list = [SPT.SPT(RuleType.RuleType.SEQUENCING), WINQ.WINQ(RuleType.RuleType.SEQUENCING), MS.MS(RuleType.RuleType.SEQUENCING), CR.CR(RuleType.RuleType.SEQUENCING)]
        self.func_selection = 0
        # action space, consists of all selectable rules
        self.output_size = len(self.func_list)
        '''
        choose the reward function for machines
        '''
        if 'reward_function' in kwargs:
            if kwargs['reward_function'] == 0:
                self.reward_function = self.get_reward0
            if kwargs['reward_function'] == 1:
                self.reward_function = self.get_reward1
        else:
            print('WARNING: reward function is not specified')
            raise Exception
        '''
        choose the architecture of state space and architecture of ANN
        and specify the address to store the trained state-dict
        needs to be specified in kwargs, otherwise abstract networks + abstract state space
        '''
        if 'MC' in kwargs and kwargs['MC']:
            print("---> Multi-Channel (MC) mode ON <---")
            self.input_size = 25 # todo: need to modify this by mengxu 2022.11.19
            # self.input_size = len(self.state_multi_channel(self.sequencing_data_generation(None)))
            self.sequencing_action_NN = network_validated(self.input_size, self.output_size)
            self.sequencing_target_NN = copy.deepcopy(self.sequencing_action_NN)
            self.dataset_name = kwargs['objective'] + "_" + str(utilLevel) + "_" + str(dueDateFactor)
            self.address_seed = "{}/sequencing_models/scenario_" + self.dataset_name + "/run_" + str(seed) + "_MC_rwd" + str(
                kwargs['reward_function']) + ".pt"  # modified by mengxu
            # self.address_seed = "{}/sequencing_models/MC_rwd"+str(kwargs['reward_function'])+".pt" #original
            self.build_state = self.state_multi_channel
            self.train = self.train_validated
            self.action_DRL = self.action_sqc_rule
            # for m in self.target_m_list:
            #     m.build_state = self.state_multi_channel
        else:
            print("WARNING: ANN TYPE NOT SPECIFIED !!!!")

        if "trained_parameter" in kwargs:  # import trained parameters for better efficiency
            # for m in self.target_m_list:
            import_address = "{}/sequencing_models/validated_" + kwargs["trained_parameter"] + ".pt"
            self.sequencing_action_NN.network.load_state_dict(torch.load(import_address.format(sys.path[0])))
            print("IMPORT FROM:", import_address)

        '''
        specify new address seed for storing the trained parameters
        '''
        if 'store_to' in kwargs:
            self.address_seed = "{}/sequencing_models/" + str(kwargs['address_seed']) + ".pt"
            print("New address seed:", self.address_seed)
        # initialize initial replay memory, a dictionary that contains empty lists of replay memory for machines
        self.rep_memo_machine = []
        # some training-related parameters
        self.minibatch_size = 64
        self.rep_memo_machine_size = 1024
        self.sequencing_action_NN_training_interval = 5  # training frequency of updating the policy network
        self.sequencing_action_NN_training_time_record = []
        self.sequencing_target_NN_update_interval = 500  # synchronize the weights of NN every 200 time units
        self.sequencing_target_NN_update_time_record = []
        # Initialize the parameters for learning of DRL
        self.discount_factor = 0.8  # how much agent care long-term rewards
        self.epsilon = 0.15  # chance of exploration
        # initialize the lists of data memory for all target machines
        # record the loss
        self.loss_time_record = []
        self.loss_record = []

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
                self.before_operation(nextEvent)
                worked = self.triggerProcessFinishEvent(nextEvent)
                # print("worked: " + str(worked))
            else:
                nextEvent.trigger(self)
            self.afterThroughput = self.throughput
            if self.throughput > self.warmupJobs and self.afterThroughput - self.beforeThroughput == 0:
                self.count = self.count + 1
            if self.count > 100000:
                self.count = 0
                self.systemState.setClockTime(np.Infinity)
                # print("bad run: self.count > 100000")
                self.eventQueue.clear()
            for workCenter in self.systemState.getWorkCenters():
                numOpsInQueue = workCenter.numOpsInQueue()
                if numOpsInQueue > 100:
                    self.systemState.setClockTime(np.Infinity)
                    self.eventQueue.clear()
        # end the training after span time
        # and store the trained parameters
        print('FINAL- replay_memory')
        print(tabulate(self.rep_memo_machine, headers=['s_t', 'a_t', 's_t+1', 'r_t']))
        print('FINAL - size of replay memory:', len(self.rep_memo_machine))
        # specify the address to store the model / state_dict
        address = self.address_seed.format(sys.path[0])
        # save the parameters of policy / action network after training
        torch.save(self.sequencing_action_NN.network.state_dict(), address)
        # after the training, print out the setting of DRL architecture
        print("Training terminated, store trained parameters to: {}".format(self.address_seed))
        self.saveLoss()

    def triggerProcessFinishEvent(self, processFinish):
        worked = False
        workCenter = processFinish.process.getWorkCenter()
        processFinish.process.getOperationOption().getJob().addProcessFinishEvent(self)
        if len(workCenter.getQueue()) > 1:
            sequencingDecisionSituation = SequencingDecisionSituation.SequencingDecisionSituation(workCenter.getQueue(), workCenter, self.getSystemState())
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
                idx_seq = int(self.numJobsArrived/period) #todo: need to double check this function
                self.func_selection = idx_seq
                selectedSequencingRule = self.func_list[idx_seq]
                self.setSequencingRule(selectedSequencingRule)
                # print("Random selected rule: " + selectedSequencingRule.name)
                s_t = self.build_state(self.sequencing_data_generation(processFinish), processFinish)
                dispatchedOp, op_idx = self.getSequencingRule().priorOperation(sequencingDecisionSituation)
                j_idx = dispatchedOp.getJob().getId() # job ID
                m_idx = workCenter.getId()
                # print('state:',self.env.now,s_t)
                # action is index of rule, NOT index or position of job
                a_t = torch.tensor(self.func_selection)
                # add the state at sequencing to job creator's corresponding repository
                workCenter.removeFromQueue(dispatchedOp)
                self.build_experience(j_idx, m_idx, s_t, a_t)
                self.complete_experience(processFinish, op_idx)
                processStartTime = np.maximum(workCenter.getReadyTime(), processFinish.time)
                nextP = Process.Process(workCenter, processFinish.process.getMachineId(), dispatchedOp, processStartTime)
                self.addEvent(ProcessStartEvent.ProcessStartEvent(nextP))
                worked = True
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
                idx_seq,s_t = self.action_random_exploration(self.sequencing_data_generation(processFinish), processFinish)
                selectedSequencingRule = self.func_list[idx_seq]
                self.setSequencingRule(selectedSequencingRule)
                # print("Random selected rule: " + selectedSequencingRule.name)
                dispatchedOp, op_idx = self.getSequencingRule().priorOperation(sequencingDecisionSituation)
                workCenter.removeFromQueue(dispatchedOp)
                # the decision is made by either of the available sequencing rule
                j_idx = dispatchedOp.getJob().getId()  # job ID
                m_idx = workCenter.getId()
                a_t = idx_seq
                # add the state at sequencing to job creator's corresponding repository
                self.build_experience(j_idx, m_idx, s_t,
                                      a_t)  # todo: need to check if using the number of arrived jobs as unit time is right??
                self.complete_experience(processFinish, op_idx)
                processStartTime = np.maximum(workCenter.getReadyTime(), processFinish.time)
                nextP = Process.Process(workCenter, processFinish.process.getMachineId(), dispatchedOp, processStartTime)
                self.addEvent(ProcessStartEvent.ProcessStartEvent(nextP))
                worked = True

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
                for i in range(10):
                    self.train()
                self.initialised = True
            if self.getEnvTime() < self.numJobsRecorded and self.numJobsArrived >= self.warmupJobs:
                if self.numJobsArrived%self.sequencing_action_NN_training_interval == 0:
                    # periodic training
                    self.train()
                # hand over the target machines' sequencing function to policy network
                idx_seq, s_t = self.action_sqc_rule(self.sequencing_data_generation(processFinish), processFinish)
                selectedSequencingRule = self.func_list[idx_seq]
                self.setSequencingRule(selectedSequencingRule)
                dispatchedOp, op_idx = self.getSequencingRule().priorOperation(sequencingDecisionSituation)
                workCenter.removeFromQueue(dispatchedOp)
                # the decision is made by either of the available sequencing rule
                j_idx = dispatchedOp.getJob().getId()  # job ID
                m_idx = workCenter.getId()
                a_t = idx_seq
                # add the state at sequencing to job creator's corresponding repository
                self.build_experience(j_idx, m_idx, s_t,
                                      a_t)  # todo: need to check if using the number of arrived jobs as unit time is right??
                self.complete_experience(processFinish, op_idx)
                processStartTime = np.maximum(workCenter.getReadyTime(), processFinish.time)
                nextP = Process.Process(workCenter, processFinish.process.getMachineId(), dispatchedOp, processStartTime)
                self.addEvent(ProcessStartEvent.ProcessStartEvent(nextP))
                # update the replay memory periodically
                if self.numJobsArrived % (self.sequencing_action_NN_training_interval*10) == 0:
                    for workCenter in self.systemState.getWorkCenters():
                        m_idx = workCenter.getId()
                        # add new memoery from corresponding rep_memo from job creator
                        self.rep_memo_machine += self.rep_memo_job[m_idx].copy()
                        # and assign top priority to new experiences
                        self.rep_memo_TDerror = torch.cat(
                            [self.rep_memo_TDerror, torch.ones(len(self.rep_memo_job[m_idx]), dtype=torch.float)])
                        # and clear the replay memory in job creator, keep it updated
                        self.rep_memo_job[m_idx] = []
                    # clear the obsolete experience periodically
                    if len(self.rep_memo_machine) > self.rep_memo_machine_size:
                        truncation = len(self.rep_memo_machine) - self.rep_memo_machine_size
                        self.rep_memo_machine = self.rep_memo_machine[truncation:]
                        self.rep_memo_TDerror = self.rep_memo_TDerror[truncation:]
                # synchronize the parameter of policy and target network
                if self.numJobsArrived % self.sequencing_target_NN_update_interval == 0:
                    self.sequencing_target_NN = copy.deepcopy(self.sequencing_action_NN)
                    print('--------------------------------------------------------')
                    print('the target network and epsilion are updated when the arrived job number is %s' % self.numJobsArrived)
                    print('--------------------------------------------------------')
                worked = True
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
        return worked



    def build_sqc_experience_repository_job(self):
        # "grand" dictionary for replay memory
        self.incomplete_rep_memo_job = {}
        self.rep_memo_job = {}
        # for each machine to be controlled, build a sub-dictionary
        # because incomplete experience must be indexed by job's index
        for workCenter in self.systemState.getWorkCenters():
            m_idx = workCenter.getId()
            self.incomplete_rep_memo_job[m_idx] = {}
            self.rep_memo_job[m_idx] = []

    '''
    3. downwards are functions used for building / updating replay memory
    '''

    def build_initial_rep_memo_parameter_sharing(self):
        for workCenter in self.getSystemState().getWorkCenters():
            m_idx = workCenter.getId()
            self.rep_memo_machine += self.rep_memo_job[m_idx].copy()
            # todo: need to double check if this is right by mengxu 2022.11.18
            #       the following is the original function
            #       these two are different and need to double check the code!!!
            # self.rep_memo += self.job_creator.rep_memo[m.m_idx].copy()
            # and clear the replay memory in job creator, keep it updated
            self.rep_memo_job[m_idx] = []
        # and the initial dummy TDerror
        self.rep_memo_TDerror = torch.ones(len(self.rep_memo_machine), dtype=torch.float)
        print('INITIALIZATION - replay_memory')
        print(tabulate(self.rep_memo_machine, headers=['s_t', 'a_t', 's_t+1', 'r_t']))
        print('INITIALIZATION - size of replay memory:', len(self.rep_memo_machine))
        print('---------------------------initialization accomplished-----------------------------')

    def getCompletionRate(self):
        sumCompletionRate = len(self.getSystemState().getJobsCompleted())
        for job in self.getSystemState().getJobsInSystem():
            sumCompletionRate += job.getCompletionRate()
        return sumCompletionRate/(len(self.getSystemState().getJobsCompleted())+len(self.getSystemState().getJobsInSystem()))

    # for those not controlled by brain, use default FIFO rule for sequencing
    def action_default(self, op, workCenter, systemState):
        return "FCFS"

    # used in the first half phase of warm up
    def action_warm_up(self, sqc_data, processFinish):
        s_t = self.build_state(sqc_data, processFinish)
        m_idx = sqc_data[-1]
        # print('state:',self.env.now,s_t)
        # action is index of rule, NOT index or position of job
        a_t = torch.tensor(self.func_selection)
        return a_t,s_t
        # # the decision is made by either of the available sequencing rule
        # job_position = self.func_list[self.func_selection](sqc_data)
        # j_idx = sqc_data[-2][job_position]
        # # add the state at sequencing to job creator's corresponding repository
        # self.build_experience(j_idx, m_idx, s_t, a_t)
        # return job_position

    # used in the second phase of warm up
    def action_random_exploration(self, sqc_data, processFinish):
        s_t = self.build_state(sqc_data, processFinish)
        m_idx = sqc_data[-1]
        # print('state:',self.env.now,s_t)
        # action is index of rule, NOT index or position of job
        self.func_selection = np.random.randint(self.output_size)
        a_t = torch.tensor(self.func_selection)
        return a_t,s_t
        # # the decision is made by either of the available sequencing rule
        # job_position = self.func_list[a_t](sqc_data)
        # j_idx = sqc_data[-2][job_position] # I think this is the id of job
        # # add the state at sequencing to job creator's corresponding repository
        # self.build_experience(j_idx, m_idx, s_t, a_t)
        # return job_position

    def action_sqc_rule(self, sqc_data, processFinish):
        s_t = self.build_state(sqc_data, processFinish)
        m_idx = sqc_data[-1]
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            a_t = torch.randint(0, self.output_size, [])
            # print('Random Action / By Brain')
        else:
            # input state to policy network, produce the state-action value
            value = self.sequencing_action_NN.forward(s_t.reshape([1, 1, self.input_size]), m_idx)
            # greedy policy
            a_t = torch.argmax(value)
            # print("State is:", s_t)
            # print('State-Action Values:', value)
            # print('Sequencing NN, action %s / By Brain'%(a_t))
        return a_t,s_t
        # the decision is made by one of the available sequencing rule
        # job_position = self.func_list[a_t](sqc_data)
        # j_idx = sqc_data[-2][job_position]
        # # add the state at sequencing to job creator's corresponding repository
        # self.build_experience(j_idx, m_idx, s_t, a_t)
        # return job_position

    # give out the information related to sequencing decision, modified by mengxu
    def sequencing_data_generation(self, processFinishEvent):
        # todo: the data used for selecting an operation from the queue by sequencing rule
        # modified by mengxu
        if processFinishEvent is None:
            workCenter = self.systemState.workCenters[0] # for initialise the CNN network
        else:
            workCenter = processFinishEvent.getProcess().getWorkCenter()
        operationOptions = workCenter.getQueue()
        current_pt = np.array([op.getProcTime() for op in operationOptions])
        remaining_job_pt = np.array([op.getWorkRemaining() for op in operationOptions])
        due_list = np.array([op.getJob().getDueDate() for op in operationOptions])
        clock_time = self.systemState.getClockTime()
        completion_rate = np.array([op.getJob().getCompletionRate() for op in operationOptions])
        time_till_due = np.array(due_list) - clock_time
        slack = np.array([op.getJob().getDueDate() - clock_time - op.getWorkRemaining() for op in operationOptions])
        winq = np.array([np.array(op.getOperation().getAverageWIQForRemainingOperations()) for op in operationOptions]) # todo: need to double check
        avlm = np.array([np.array(op.getOperation().getAverageMachineWaitingTimeForRemainingOperations(clock_time)) for op in operationOptions]) # todo: need to double check
        next_pt = np.array([op.getNextProcTime() for op in operationOptions])
        remaining_no_op = np.array([op.getNumOpsRemaining() for op in operationOptions])
        waited_time = np.array([clock_time - op.getJob().getReleaseTime() for op in operationOptions]) # todo: need to double check: I think this is time in queue or time in system
        wc_idx = workCenter.getId()
        queue = workCenter.getQueue()
        m_idx = workCenter.getId() # in my DFJSS simulator a wc has a machine, so I think this is the same

        self.sequencing_data = [current_pt, remaining_job_pt, np.array(due_list), clock_time,
             completion_rate, time_till_due, slack, winq, avlm, next_pt, remaining_no_op,
             waited_time, wc_idx, queue, m_idx]
        # print(self.sequencing_data)
        return self.sequencing_data



    # add the experience to job creator's incomplete experiece memory
    def build_experience(self, j_idx, m_idx, s_t, a_t):
        self.incomplete_rep_memo_job[m_idx][self.getNumJobsArrived()] = [s_t, a_t]

    '''
        2. downwards are functions used for building the state of the experience (replay memory)
        '''
    '''
    local data consists of:
    0            1                 2         3        4
    [current_pt, remaining_job_pt, due_list, env.now, completion_rate,
    5              6      7     8     9        10         11           12/-3   13/-2  14/-1
    time_till_due, slack, winq, avlm, next_pt, rem_no_op, waited_time, wc_idx, queue, m_idx]
    '''

    # this function is only called when sequencing point arrives that is the job arrival event and process finish event! by meng
    def state_multi_channel(self, sqc_data,
                            event):  # I think this should be called when JobArrivalEvent happen!!! by mengxu
        # information in job number, global and local
        in_system_job_no = self.getNumJobsArrived()
        local_job_no = len(sqc_data[0])
        # the information of coming job (currently being processed by other machines)
        # print('coming jobs:',self.job_creator.next_wc_list, self.job_creator.arriving_job_slack_list, self.job_creator.release_time_list, sqc_data[-3])
        if isinstance(event, JobArrivalEvent.JobArrivalEvent):
            arriving_job_no = 1
            arriving_job_time = event.job.getReleaseTime() - self.getClockTime()  # average time from now when next job arrives at workcenter
            workRemaining = 0
            for operation in event.job.operation_list:
                workRemaining += operation.getMeanProcessTime()
            arriving_job_slack = event.job.getDueDate() - self.getClockTime() - workRemaining  # what's the average slack time of the arriving job
        elif isinstance(event, ProcessFinishEvent.ProcessFinishEvent):
            arriving_job_no = 0
            arriving_job_time = 0
            arriving_job_slack = 0
        else:
            print("Error! only the above two situations can happen!")
        # print(arriving_jobs, arriving_job_no, arriving_job_time, arriving_job_slack, self.env.now, sqc_data[-3])
        # information of progression of jobs, get from the job creator
        global_comp_rate = self.getCompletionRate()
        global_realized_tard_rate = self.systemState.getRealTardyRate()
        # todo: need to double check the global_realized_tard_rate and global_exp_tard_rate is a list or a number by mengxu 2022.11.18
        global_exp_tard_rate = self.systemState.getExpTardyRate()

        available_time = []
        for workCenter in self.getSystemState().getWorkCenters():
            available_time.append(np.maximum(0, workCenter.getReadyTime() - self.getClockTime()))
        # get the pt of all remaining jobs in system
        rem_pt = [None for i in range(self.getNumJobsArrived())]
        # need loop here because remaining_pt have different length
        for job in self.getSystemState().getJobsInSystem():
            rem_pt[int(job.getId())] = job.getRemainingProcessingTime()
        for job in self.getSystemState().getJobsCompleted():
            rem_pt[int(job.getId())] = 0
        for job in self.getSystemState().warmupJobsCompleted:
            rem_pt[int(job.getId())] = 0
        # processing time related data
        pt_share = available_time[sqc_data[-1]] / sum(available_time)  # sum of pt / sum of available time
        global_pt_CV = np.std(rem_pt) / np.mean(rem_pt)
        # information of queuing jobs in queue
        local_pt_sum = np.sum(sqc_data[0])
        local_pt_mean = np.mean(sqc_data[0])
        local_pt_min = np.min(sqc_data[0])
        local_pt_CV = np.std(sqc_data[0]) / local_pt_mean
        # information of queuing jobs in remaining processing time
        local_remaining_pt_sum = np.sum(sqc_data[1])
        local_remaining_pt_mean = np.mean(sqc_data[1])
        local_remaining_pt_max = np.max(sqc_data[1])
        local_remaining_pt_CV = np.std(sqc_data[1]) / local_remaining_pt_mean
        # information of WINQ
        avlm_mean = np.mean(sqc_data[8])
        avlm_min = np.min(sqc_data[8])
        avlm_CV = np.std(sqc_data[8]) / avlm_mean
        # time-till-due related data:
        time_till_due = sqc_data[5]
        realized_tard_rate = time_till_due[time_till_due < 0].size / local_job_no  # ratio of tardy jobs
        ttd_sum = time_till_due.sum()
        ttd_mean = time_till_due.mean()
        ttd_min = time_till_due.min()
        ttd_CV = (time_till_due.std() / ttd_mean).clip(-2, 2)
        # slack-related data:
        slack = sqc_data[6]
        exp_tard_rate = slack[slack < 0].size / local_job_no  # ratio of jobs expect to be tardy
        slack_sum = slack.sum()
        slack_mean = slack.mean()
        slack_min = slack.min()
        slack_CV = (slack.std() / slack_mean).clip(-2, 2)
        # use raw data, and leave the magnitude adjustment to normalization layers
        no_info = [in_system_job_no, arriving_job_no, local_job_no]  # info in job number
        pt_info = [local_pt_sum, local_pt_mean, local_pt_min]  # info in processing time
        remaining_pt_info = [local_remaining_pt_sum, local_remaining_pt_mean, local_remaining_pt_max, avlm_mean,
                             avlm_min]  # info in remaining processing time
        ttd_slack_info = [ttd_mean, ttd_min, slack_mean, slack_min, arriving_job_slack]  # info in time till due
        progression = [pt_share, global_comp_rate, global_realized_tard_rate, global_exp_tard_rate]  # progression info
        heterogeneity = [global_pt_CV, local_pt_CV, ttd_CV, slack_CV, avlm_CV]  # heterogeneity
        # concatenate the data input
        s_t = np.nan_to_num(
            np.concatenate([no_info, pt_info, remaining_pt_info, ttd_slack_info, progression, heterogeneity]), nan=0,
            posinf=1, neginf=-1)
        # convert to tensor
        s_t = torch.tensor(s_t, dtype=torch.float)
        return s_t

    def train_validated(self):
        """
        draw the random minibatch to train the network
        every element in the replay menory is a list [s_0, a_0, s_1, r_0]
        all element of this list are tensors
        """
        # todo: need to check why self.rep_memo_machine could be Nan??? 2022.11.22
        # size = min(1, self.minibatch_size)
        size = min(len(self.rep_memo_machine),self.minibatch_size)
        minibatch = random.sample(self.rep_memo_machine,size)
        '''
        slice, and stack the 1D tensors to several 3D tensors (batch, channel, vector)
        the "torch.stack" is only applicable when the augment is a list of tensors, or multi-dimensional tensor
        '''
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        sample_s1_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1)
        '''
        the size of these batches:
        sample_s0_batch = sample_s1_batch = minibatch size * 1 * input_size
        sample_a0_batch = sample_r0_batch = minibatch size * m_no
        sample_r0_batch = minibatch size
        '''
        # get the Q value (current value of state-action pair) of s0
        Q_0 = self.sequencing_action_NN.forward(sample_s0_batch)
        #print('Q_0 is:\n', Q_0)
        #print('a_0 is:', sample_a0_batch)
        # get the current state-action value of actions that would have been taken
        current_value = Q_0.gather(1, sample_a0_batch)
        #print('current value is:', current_value)
        '''
        get the Q Value of s_1 in both action and target network, to estimate the state value
        architecture is DDQN, NOT DQN !!!
        evaluate the greedy policy according to action network, but using the target network to estimate the value
        '''
        Q_1_action = self.sequencing_action_NN.forward(sample_s1_batch).detach()
        Q_1_target = self.sequencing_target_NN.forward(sample_s1_batch).detach()
        #print('Q_1_action is:\n', Q_1_action)
        #print('Q_1_target is:\n', Q_1_target)
        '''
        size of Q_0, Q_1_action and Q_1_target = minibatch size * m_no
        they're 2D tensors
        '''
        max_Q_1_action, max_Q_1_action_idx = torch.max(Q_1_action, dim=1) # use action network to get action, rather than max operation
        #print('max value of Q_1_action is:\n', max_Q_1_action)
        max_Q_1_action_idx = max_Q_1_action_idx.reshape([size,1])
        #print('max idx of Q_1_action is:\n', max_Q_1_action_idx)
        # adjust the max_Q of s_0 by the discount factor (refer to Bellman Equation and TD)
        next_state_value = Q_1_target.gather(1, max_Q_1_action_idx)
        #print('estimated value of next state is:\n', next_state_value)
        next_state_value *= self.discount_factor
        #print('discounted next state value is:\n', next_state_value)
        '''
        the sum of reward and discounted max_Q is the target value
        target value is 2D matrix, size = minibatch_size * m_no
        '''
        #print('reward batch is:', sample_r0_batch)
        target_value = (sample_r0_batch + next_state_value)
        #print('target value is:', target_value)
        #print('TD error:',target_value - current_value)
        # calculate the loss
        loss = self.sequencing_action_NN.loss_func(current_value, target_value)
        self.loss_time_record.append(self.getNumJobsArrived())
        self.loss_record.append(float(loss))
        if not self.getNumJobsArrived()%50:
            print('Time: %s, loss: %s:'%(self.getNumJobsArrived(), loss))
        # first, clear the gradient (old) of parameters
        self.sequencing_action_NN.optimizer.zero_grad()
        # second, calculate gradient (new) of parameters
        loss.backward(retain_graph=True)
        '''
        # check the gradient, to avoid exploding/vanishing gradient, very seldom though
        for param in self.sequencing_action_NN.module_dict[m_idx].parameters():
            print(param.grad.norm())
        '''
        # third, update the parameters
        self.sequencing_action_NN.optimizer.step()

    '''
    4. downwards are functions related to the calculation of reward and construction of state
       only be called if the sequencing learning mode is activated
       the options of reward function are listed at bottom
    '''

    # this function is called only if self.sequencing_learning_event is triggered
    # when this function is called upon the completion of an operation
    # it add received data to corresponding record in job creator's incomplete_rep_memo
    def complete_experience(self, processFinishEvent, op_idx):
        # it's possible that not all machines keep memory for learning
        # machine that needs to keep memory don't keep record for all jobs
        # only when they have to choose from several queuing jobs
        try:
            workCenter = processFinishEvent.getProcess().getWorkCenter()
            decision_point = self.numJobsArrived
            #todo: neeed to check the meaning of decision_point in original code
            # check whether corresponding experience exists, if not, ends at this line
            # print("self.incomplete_rep_memo_job[int(workCenter.getId())][int(decision_point)]: " + str(self.incomplete_rep_memo_job[int(workCenter.getId())][int(decision_point)]))
            self.incomplete_rep_memo_job[int(workCenter.getId())][int(decision_point)]
            # if not ref:
            #     dispatchedOp = workCenter.getQueue()[op_idx]
            #     workCenter.removeFromQueue(dispatchedOp)
            #     pass
            # print('PARAMETERS',self.m_idx,self.decision_point,self.env.now)
            # print('BEFORE\n',self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point])
            # if yes, get the global state
            local_data = self.sequencing_data_generation(processFinishEvent)
            s_t = self.build_state(local_data, processFinishEvent)
            # print(self.m_idx,s_t)
            r_t = self.reward_function(processFinishEvent, op_idx)  # can change the reward function, by sepecifying before the training
            # print(self.env.now, r_t)
            self.sqc_reward_record_job.append([self.numJobsArrived, r_t])
            self.incomplete_rep_memo_job[int(workCenter.getId())][int(decision_point)] += [s_t, r_t]
            # print(self.job_creator.incomplete_rep_memo[self.m_idx])
            # print(self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point])
            complete_exp = self.incomplete_rep_memo_job[int(workCenter.getId())].pop(int(decision_point))
            # and add it to rep_memo
            self.rep_memo_job[int(workCenter.getId())].append(complete_exp)
            # print(self.job_creator.rep_memo[self.m_idx])
            # print('AFTER\n',self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point])
            # print(self.m_idx,self.env.now,'state: ',s_t,'reward: ',r_t)
        except:
            pass

    #todo: need to modify this part 2022.11.22
    # testing reward function, check if the agent learns, this function encourages using SPT
    def get_reward0(self, processFinishEvent, op_idx):
        # before_op_slack, id_job, current_pt, waiting_jobs, pt_chosen, before_op_winq_loser, before_op_winq_chosen = self.before_operation(
        #     processFinishEvent, op_idx)
        workCenter = processFinishEvent.getProcess().getWorkCenter()
        operationOptions = workCenter.getQueue()
        # id_job = operationOptions[op_idx].getJob().getId()
        id_job = op_idx
        self.pt_chosen = self.current_pt[id_job]
        if self.pt_chosen <= self.current_pt[:self.waiting_jobs - 1].mean():
            r_t = 1
        else:
            r_t = 0
        r_t = torch.tensor(r_t, dtype=torch.float)
        # print(self.numJobsArrived, 'reward_torch:', r_t)
        return r_t

    # those functions are called only if self.sequencing_learning_event is triggered
    # called only upon the completion of all operations of a job
    # it calculates the reward for all machines that job went through
    # then a complete experience is constructed for learning
    # change the parameters to achieve better performance in different scenarios

    # example of a real reward function
    def get_reward1(self, processFinishEvent, op_idx):# todo: need to double check by Mengxu 2022.11.22
        # before_op_winq_loser, before_op_winq_chosen = self.before_operation(processFinishEvent, op_idx)
        printRef = False
        workCenter = processFinishEvent.getProcess().getWorkCenter()
        operationOptions = workCenter.getQueue()
        # id_job = operationOptions[op_idx].getJob().getId()
        id_job = op_idx
        self.pt_chosen = self.current_pt[id_job]
        if printRef:
            print("self.pt_chosen: " + str(self.pt_chosen))
        self.before_op_winq_chosen = self.winq[id_job]
        if printRef:
            print("self.before_op_winq_chosen: " + str(self.before_op_winq_chosen))
        self.before_op_winq_loser = np.delete(self.winq, id_job)
        if printRef:
            print("self.before_op_winq_loser: " + str(self.before_op_winq_loser))

        slack = self.before_op_slack
        if printRef:
            print("slack: " + str(slack))
        critical_level = 1 - slack / (np.absolute(slack) + 50)
        if printRef:
            print("critical_level: " + str(critical_level))
        # get critical level for jobs, chosen and loser, respectively
        critical_level_chosen = critical_level[id_job]
        if printRef:
            print("critical_level_chosen: " + str(critical_level_chosen))
        critical_level_loser = np.delete(critical_level, id_job)  # could be a vector or scalar
        if printRef:
            print("critical_level_loser: " + str(critical_level_loser))
        # calculate adjusted earned slack for the chosen job
        earned_slack_chosen = np.mean(self.current_pt[:self.waiting_jobs - 1])
        if printRef:
            print("earned_slack_chosen: " + str(earned_slack_chosen))
        earned_slack_chosen *= critical_level_chosen
        if printRef:
            print("earned_slack_chosen: " + str(earned_slack_chosen))
        # calculate the AVERAGE adjusted slack consumption for jobs that not been chosen
        consumed_slack_loser = self.pt_chosen * critical_level_loser.mean()
        if printRef:
            print("consumed_slack_loser: " + str(consumed_slack_loser))
        # slack reward
        rwd_slack = earned_slack_chosen - consumed_slack_loser
        if printRef:
            print("rwd_slack: " + str(rwd_slack))
        # WINQ reward
        rwd_winq = (self.before_op_winq_loser.mean() - self.before_op_winq_chosen) * 0.2
        if printRef:
            print("rwd_winq: " + str(rwd_winq))
        # calculate the reward
        # print(rwd_slack, rwd_winq)
        rwd = ((rwd_slack + rwd_winq) / 20).clip(-1, 1)
        if printRef:
            print("rwd: " + str(rwd))
        # optional printout
        # print(self.env.now,'slack and pt:', slack, critical_level, self.position, self.pt_chosen, self.current_pt[:self.waiting_jobs-1])
        # print(self.env.now,'winq and reward:',self.before_op_winq_chosen, self.before_op_winq_loser, earned_slack_chosen, consumed_slack_loser)
        # print(self.numJobsArrived,'reward:',rwd) # todo: should check why reward can be nan??? 2022.11.22
        r_t = torch.tensor(rwd, dtype=torch.float)
        # print(self.numJobsArrived, 'reward_torch:', r_t)
        return r_t

    def saveLoss(self):
        fileName = './sequencing_models/scenario_' + str(self.dataset_name) + '/' + str(self.run_seed) + '_training_loss'
        np.save(fileName, self.loss_record)


    # update information that will be used for calculating the rewards
    def before_operation(self, processFinishEvent):
        workCenter = processFinishEvent.getProcess().getWorkCenter()
        operationOptions = workCenter.getQueue()
        # id_job = operationOptions[op_idx].getJob().getId()
        # number of jobs that to be sequenced, and their ttd and slack
        self.waiting_jobs = len(operationOptions)
        due_list = np.array([op.getJob().getDueDate() for op in operationOptions])
        time_till_due = np.array(due_list) - self.systemState.getClockTime()
        before_op_ttd = time_till_due
        # before_op_ttd_chosen = before_op_ttd[id_job]  # I think self.position means the id of job
        # before_op_ttd_loser = np.delete(before_op_ttd, id_job)
        tardy_jobs = len(time_till_due[time_till_due < 0])
        # self.before_op_realized_tard_rate =tardy_jobs/len(self.queue)
        # print('before realized tard rate: ', self.before_op_realized_tard_rate)
        # op.getJob().getDueDate() - clock_time - op.getWorkRemaining()
        # initial_slack = operationOptions[op_idx].getJob().getDueDate() - self.systemState.getClockTime() - (workCenter.getReadyTime() - operationOptions[op_idx].getReadyTime()) - operationOptions[op_idx].getWorkRemaining()
        # todo: I think this store the estimated slack time when the job just arrive at this workcenter, need to modify this
        remaining_job_pt = np.array([op.getWorkRemaining() for op in operationOptions])
        self.current_pt = np.array([op.getProcTime() for op in operationOptions])
        before_op_remaining_pt = remaining_job_pt + self.current_pt
        # before_op_remaining_pt_chosen = before_op_remaining_pt[id_job]
        # before_op_remaining_pt_loser = np.delete(before_op_remaining_pt, id_job)
        self.current_slack = time_till_due - before_op_remaining_pt
        # exp_tardy_jobs = len(current_slack[current_slack < 0])
        # get information of all jobs before operation
        # before_op_exp_tard = current_slack[current_slack < 0]
        # before_op_sum_exp_tard = before_op_exp_tard.sum()
        self.before_op_slack = self.current_slack
        # before_op_sum_slack = self.before_op_slack.sum()
        # calculate the critical level  of all queuing jobs
        # critical_level = 1 - current_slack / 100
        # critical_level_chosen = critical_level[id_job]
        # print(current_slack, self.critical_level,self.critical_level_chosen)
        # get the information of the selected job
        # self.pt_chosen = self.current_pt[id_job]
        # initial_slack_chosen = initial_slack[id_job]
        # initial_slack_chosen = operationOptions[op_idx].getJob().getDueDate() - self.systemState.getClockTime() - (
        #         workCenter.getReadyTime() - operationOptions[op_idx].getReadyTime()) - operationOptions[
        #                            op_idx].getWorkRemaining()
        # before_op_slack_chosen = current_slack[id_job]
        # before_op_exp_tard_chosen = min(0, before_op_slack_chosen)
        self.winq = np.array(
            [np.array(op.getOperation().getAverageWIQForRemainingOperations()) for op in operationOptions])
        # self.before_op_winq_chosen = winq[id_job]
        # get the information of jobs that haven't been selected (loser)
        # before_op_slack_loser = np.delete(current_slack, id_job)  # those haven't been selected
        # critical_level_loser = np.delete(critical_level, id_job)
        # before_op_sum_exp_tard_loser = before_op_slack_loser[before_op_slack_loser < 0].sum()
        # before_op_sum_slack_loser = before_op_slack_loser.sum()
        # self.before_op_winq_loser = np.delete(winq, id_job)
        # return self.before_op_slack, id_job, self.current_pt, self.waiting_jobs, self.pt_chosen, self.before_op_winq_loser, self.before_op_winq_chosen


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
