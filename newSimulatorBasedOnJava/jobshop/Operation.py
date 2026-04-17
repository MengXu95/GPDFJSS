import numpy
from numpy import sort, mean
from newSimulatorBasedOnJava.simulation import RoutingDecisionSituation


class Operation:
    # def __init__(self, job, id, **kwargs):
    #     self.job = job
    #     self.id = id
    #     self.operationOptions = []

    def __init__(self, job, id, **kwargs):
        self.job = job
        self.id = id
        self.operationOptions = []
        self.next = None
        self.selectedWorkCenter = None
        self.uploadTime = -1
        self.transferTime = -1
        self.downloadTime = -1
        self.completed = False

    def isCompleted(self):
        return self.completed

    def getJob(self):
        return self.job

    def getId(self):
        return self.id

    def setNext(self, next):
        self.next = next

    def getNext(self):
        return self.next

    def getOperationOptions(self):
        return self.operationOptions

    def addOperationOption(self,option):
        self.operationOptions.append(option)

    def setSelectedWorkCenter(self, selectedWorkCenter):
        self.selectedWorkCenter = selectedWorkCenter

    def getSelectedWorkCenter(self):
        return self.selectedWorkCenter

    def setUploadTimeToSelectedWorkCenter(self, uploadTime):
        self.uploadTime = uploadTime

    def setTransferTimeToSelectedWorkCenter(self, transferTime):
        self.transferTime = transferTime

    def setDownloadTimeToSelectedWorkCenter(self, downloadTime):
        self.downloadTime = downloadTime

    def getUploadTime(self):
        return self.uploadTime

    def getDownloadTime(self):
        return self.downloadTime

    def getTransferTime(self):
        return self.transferTime

    def getLeastWorkLoad(self):
        leastWorkLoad = self.getOperationOptions()[0].getWorkCenter().getWorkInQueue()
        for j in range(0, len(self.operationOptions)):
            if self.getOperationOptions()[j].getWorkCenter().getWorkInQueue() < leastWorkLoad:
                leastWorkLoad = self.getOperationOptions()[j].getWorkCenter().getWorkInQueue()
        return leastWorkLoad

    def getMaxWorkLoad(self):
        maxWorkLoad = self.getOperationOptions()[0].getWorkCenter().getWorkInQueue()
        for j in range(0, len(self.operationOptions)):
            if self.getOperationOptions()[j].getWorkCenter().getWorkInQueue() > maxWorkLoad:
                maxWorkLoad = self.getOperationOptions()[j].getWorkCenter().getWorkInQueue()
        return maxWorkLoad

    def getAveWorkLoad(self):
        totalWorkLoad = 0
        for j in range(0, len(self.operationOptions)):
            totalWorkLoad = totalWorkLoad + self.getOperationOptions()[j].getWorkCenter().getWorkInQueue()
        return totalWorkLoad/len(self.operationOptions)

    def getLeastNumOfOperation(self):
        leastNumOfOperation = self.getOperationOptions()[0].getWorkCenter().getNumOpsInQueue()
        for j in range(0, len(self.operationOptions)):
            if self.getOperationOptions()[j].getWorkCenter().getNumOpsInQueue() < leastNumOfOperation:
                leastNumOfOperation = self.getOperationOptions()[j].getWorkCenter().getNumOpsInQueue()
        return leastNumOfOperation

    def getMaxNumOfOperation(self):
        maxNumOfOperation = self.getOperationOptions()[0].getWorkCenter().getNumOpsInQueue()
        for j in range(0, len(self.operationOptions)):
            if self.getOperationOptions()[j].getWorkCenter().getNumOpsInQueue() > maxNumOfOperation:
                maxNumOfOperation = self.getOperationOptions()[j].getWorkCenter().getNumOpsInQueue()
        return maxNumOfOperation

    def getAveNumOfOperation(self):
        totalNumOfOperation = self.getOperationOptions()[0].getWorkCenter().getNumOpsInQueue()
        for j in range(0, len(self.operationOptions)):
            totalNumOfOperation = totalNumOfOperation + self.getOperationOptions()[j].getWorkCenter().getNumOpsInQueue()
        return totalNumOfOperation/len(self.operationOptions)

    def getLeastProcessTime(self):
        leastProcessTime = self.getOperationOptions()[0].getProcTime()
        for j in range(0, len(self.operationOptions)):
            if self.getOperationOptions()[j].getProcTime() < leastProcessTime:
                leastProcessTime = self.getOperationOptions()[j].getProcTime()
        return leastProcessTime

    def getMaxProcessTime(self):
        maxProcessTime = self.getOperationOptions()[0].getProcTime()
        for j in range(0, len(self.operationOptions)):
            if self.getOperationOptions()[j].getProcTime() > maxProcessTime:
                maxProcessTime = self.getOperationOptions()[j].getProcTime()
        return maxProcessTime

    def getMedianProcessTime(self):
        totalProcessTime = []
        for j in range(0, len(self.operationOptions)):
            totalProcessTime.append(self.getOperationOptions()[j].getProcTime())
        totalProcessTime = sort(totalProcessTime)
        if len(totalProcessTime) % 2 == 0:
            median = (totalProcessTime[int(len(totalProcessTime) / 2)] + totalProcessTime[
                int(len(totalProcessTime) / 2) - 1]) / 2
        else:
            median = totalProcessTime[int(len(totalProcessTime) / 2)]
        return median

    def getMedian(self, totalProcessTime):
        totalProcessTime = sort(totalProcessTime)
        if len(totalProcessTime)%2==0:
            median = (totalProcessTime[int(len(totalProcessTime)/2)] + totalProcessTime[int(len(totalProcessTime)/2)-1])/2
        else:
            median = totalProcessTime[int(len(totalProcessTime)/2)]
        return median

    def getOperationOption(self):
        highestProcTime = -numpy.inf
        best = None
        for option in self.operationOptions:
            if option.getProcTime() > highestProcTime or highestProcTime == -numpy.inf:
                highestProcTime = option.getProcTime()
                best = option
        return best

    def chooseOperationOption(self, systemState, routingRule):
        decisionSituation = self.routingDecisionSituation(systemState)
        if routingRule is None:
            print("error: routing rule is null!!!")
        return routingRule.nextOperationOption(decisionSituation)

    def routingDecisionSituation(self, systemState):
        return RoutingDecisionSituation.RoutingDecisionSituation(self.operationOptions,systemState)

    def getMeanProcessTime(self):
        sumProcessTime = 0
        for taskOption in self.operationOptions:
            sumProcessTime = sumProcessTime + taskOption.getProcTime()
        return sumProcessTime/len(self.operationOptions)

    def getAverageWIQForRemainingOperations(self): # for RL
        operation = self.next
        if operation is None:
            return 0
        averageWIQRemaining = []
        while operation is not None:
            sumWIQeachOp = 0
            operationOptions = operation.getOperationOptions()
            for op in operationOptions:
                sumWIQeachOp += op.getWorkCenter().getWorkInQueue()
            sumWIQeachOp /= len(operationOptions)
            averageWIQRemaining.append(sumWIQeachOp)
            operation = operation.next
        return mean(averageWIQRemaining) # todo: need to double check this

    def getAverageMachineWaitingTimeForRemainingOperations(self, clockTime): # for RL
        operation = self.next
        averageMWTRemaining = []
        while operation is not None:
            sumMWTeachOp = 0
            operationOptions = operation.getOperationOptions()
            for op in operationOptions:
                sumMWTeachOp += max(0, op.getWorkCenter().getReadyTime() - clockTime) # todo: need to double check
            sumMWTeachOp /= len(operationOptions)
            averageMWTRemaining.append(sumMWTeachOp)
            operation = operation.next
        return mean(averageMWTRemaining)  # todo: need to double check this

    def getString(self):
        concatenatedString = " ".join(["job ", str(self.job.id), " operation ", str(self.id), "/", str(len(self.job.operation_list))])
        return concatenatedString

    
