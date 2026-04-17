from newSimulatorBasedOnJava.jobshop import Machine


class WorkCenter:
    def __init__(self, id, **kwargs):
        self.id = id
        self.numMachines = 1
        self.queue = []
        self.machineReadyTimes = [0 for i in range(self.numMachines)]
        self.workInQueue = 0
        self.busyTime = 0
        self.processingRate = -1

    def getId(self):
        return self.id

    def getNumMachines(self):
        return self.numMachines

    def getQueue(self):
        return self.queue

    def getProcessingRate(self):
        return self.processingRate

    def setProcessingRate(self, processingRate):
        self.processingRate = processingRate

    def getMinProcessTimeInQueue(self):
        if len(self.queue) == 0:
            return 0
        minProcessTime = self.getQueue()[0].getProcTime()
        for i in range(len(self.queue)):
            if minProcessTime > self.getQueue()[i].getProcTime():
                minProcessTime = self.getQueue()[i].getProcTime()
        return minProcessTime

    def getMaxProcessTimeInQueue(self):
        if len(self.queue) == 0:
            return 0
        maxProcessTime = self.getQueue()[0].getProcTime()
        for i in range(len(self.queue)):
            if maxProcessTime < self.getQueue()[i].getProcTime():
                maxProcessTime = self.getQueue()[i].getProcTime()
        return maxProcessTime

    def getWorkInQueue(self):
        return self.workInQueue

    def getBusyTime(self):
        return self.busyTime

    def getReadyTime(self):
        # for improve speed by mengxu 2022.11.24
        return self.machineReadyTimes[0]

        # original
        # readyTime = self.machineReadyTimes[0]
        # for i in range(len(self.machineReadyTimes)):
        #     t = self.machineReadyTimes[i]
        #     if readyTime > t:
        #         readyTime = t
        # return readyTime

    def getRemainProcessTime(self):
        # for improve speed by mengxu 2022.11.24
        return sum([i.getProcTime() for i in self.queue])
        # original
        # remainProcesstime = 0
        # for i in range(len(self.queue)):
        #     remainProcesstime = remainProcesstime + self.queue[i].getProcTime()
        # return remainProcesstime

    def setMachineReadyTime(self, idx, readyTime):
        self.machineReadyTimes[idx] = readyTime

    def numOpsInQueue(self):
        return len(self.queue)

    def reset(self):
        # for improve speed by mengxu 2022.11.24
        readyTime = 0
        self.queue.clear()
        self.machineReadyTimes = [0 for i in range(self.numMachines)]
        self.workInQueue = 0
        self.busyTime = readyTime

        # original
        # readyTime = 0
        # self.queue.clear()
        # for i in range(self.numMachines):
        #     self.machineReadyTimes[i] = readyTime
        # self.workInQueue = 0
        # self.busyTime = readyTime

    def addToQueue(self, operationOption):
        self.queue.append(operationOption)
        self.workInQueue = self.workInQueue + operationOption.getProcTime()

    def removeFromQueue(self, operationOption):
        self.queue.remove(operationOption)
        if len(self.queue) == 0:
            self.workInQueue = 0
        else:
            self.workInQueue = self.workInQueue - operationOption.getProcTime()

    def earliestReadyMachine(self):
        # for improve speed by mengxu 2022.11.24
        return Machine.Machine(0, self, self.machineReadyTimes[0])
        # original
        # earliestReadyMachine = Machine.Machine(0, self, self.machineReadyTimes[0])
        # for i in range(len(self.machineReadyTimes)):
        #     if self.machineReadyTimes[i] < earliestReadyMachine.getReadyTime():
        #         earliestReadyMachine = Machine.Machine(i, self, self.machineReadyTimes[i])
        # return earliestReadyMachine

    def incrementBusyTime(self, value):
        self.busyTime += value

    def clone(self):
        clonedQ = []
        for i in range(len(self.queue)):
            clonedQ.append(self.queue[i].clone())
        clonedMRT = []
        for i in range(len(self.machineReadyTimes)):
            clonedMRT.append(self.machineReadyTimes[i])

        workCenter = WorkCenter(self.id)
        workCenter.numMachines = 1
        workCenter.queue = clonedQ
        workCenter.machineReadyTimes = clonedMRT
        workCenter.workInQueue = self.workInQueue
        workCenter.busyTime = self.busyTime
        workCenter.processingRate = self.processingRate
        return workCenter



