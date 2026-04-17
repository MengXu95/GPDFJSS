import newSimulatorBasedOnJava.rule.AbstractRule as AbstractRule
import numpy as np

class GPRule(AbstractRule.AbstractRule):
    def __init__(self, type, gpTree, **kwargs):
        name = "GPRule"
        AbstractRule.AbstractRule.__init__(self, name, type)
        self.gpTree = gpTree

    def getGPTree(self):
        return self.gpTree

    def setGPTree(self, gpTree):
        self.gpTree = gpTree

    def priority(self, op, workCenter, systemState):
        # print(self.gpTree)
        priority,_ = self.treeNode_R(self.gpTree, 0, op, workCenter, systemState)
        return priority

    # original for test I think
    # def treeNode_R(self, tree, index, op, workCenter, systemState):
    #     # if tree[index]
    #     # print(tree)
    #     # print(tree[index].value)
    #     if tree[index] == 'add':
    #         return self.treeNode_R(tree, index + 1, op, workCenter, systemState) + self.treeNode_R(tree, index + 2, op, workCenter, systemState)
    #     elif tree[index] == 'subtract':
    #         return self.treeNode_R(tree, index + 1, op, workCenter, systemState) - self.treeNode_R(tree, index + 2, op, workCenter, systemState)
    #     elif tree[index] == 'multiply':
    #         return self.treeNode_R(tree, index + 1, op, workCenter, systemState) * self.treeNode_R(tree, index + 2, op, workCenter, systemState)
    #     elif tree[index] == 'protected_div':
    #         return self.protected_div(self.treeNode_R(tree, index + 1, op, workCenter, systemState),
    #             self.treeNode_R(tree, index + 2, op, workCenter, systemState))
    #     elif tree[index] == 'maximum':
    #         return np.maximum(self.treeNode_R(tree, index + 1, op, workCenter, systemState),
    #                           self.treeNode_R(tree, index + 2, op, workCenter, systemState))
    #     elif tree[index] == 'minimum':
    #         return np.minimum(self.treeNode_R(tree, index + 1, op, workCenter, systemState),
    #                           self.treeNode_R(tree, index + 2, op, workCenter, systemState))
    #     elif tree[index] == 'NIQ':
    #         return len(workCenter.getQueue())
    #     elif tree[index] == 'WIQ':
    #         return workCenter.getWorkInQueue()
    #     elif tree[index] == 'MWT':
    #         return systemState.getClockTime() - workCenter.getReadyTime()
    #     elif tree[index] == 'PT':
    #         return op.getProcTime()
    #     elif tree[index] == 'NPT':
    #         return op.getNextProcTime()
    #     elif tree[index] == 'OWT':
    #         return systemState.getClockTime() - op.getReadyTime()
    #     elif tree[index] == 'WKR':
    #         return op.getWorkRemaining()
    #     elif tree[index] == 'NOR':
    #         return op.getNumOpsRemaining()
    #     elif tree[index] == 'W':
    #         return op.getJob().getWeight()
    #     elif tree[index] == 'TIS':
    #         return systemState.getClockTime() - op.getJob().getReleaseTime()
    #     elif tree[index] == 'SLACK':
    #         return op.getJob().getDueDate() - systemState.getClockTime() - op.getWorkRemaining()
    #     else:
    #         print("None of the terminal! Error here!")
    #         # return -100

    # original for training
    def treeNode_R(self, tree, index, op, workCenter, systemState):
        if tree[index].arity == 2:
            left, length_left = self.treeNode_R(tree, index + 1, op, workCenter, systemState)
            right, length_right = self.treeNode_R(tree, index + length_left + 1, op, workCenter, systemState)
            if tree[index].name == 'add':
                return left + right, length_left + length_right + 1
            elif tree[index].name == 'subtract':
                return left - right, length_left + length_right + 1
            elif tree[index].name == 'multiply':
                return left * right, length_left + length_right + 1
            elif tree[index].name == 'protected_div':
                return self.protected_div(left, right), length_left + length_right + 1
            elif tree[index].name == 'maximum':
                return np.maximum(left, right), length_left + length_right + 1
            elif tree[index].name == 'minimum':
                return np.minimum(left, right), length_left + length_right + 1
        elif tree[index].arity == 0:
            if tree[index].name == 'NIQ':
                return len(workCenter.getQueue()),1
            elif tree[index].name == 'WIQ':
                return workCenter.getWorkInQueue(),1
            elif tree[index].name == 'MWT':
                return systemState.getClockTime() - workCenter.getReadyTime(),1
            elif tree[index].name == 'PT':
                return op.getProcTime(),1
            elif tree[index].name == 'NPT':
                return op.getNextProcTime(),1
            elif tree[index].name == 'OWT':
                return systemState.getClockTime() - op.getReadyTime(),1
            elif tree[index].name == 'WKR':
                return op.getWorkRemaining(),1
            elif tree[index].name == 'NOR':
                return op.getNumOpsRemaining(),1
            elif tree[index].name == 'W':
                return op.getJob().getWeight(),1
            elif tree[index].name == 'TIS':
                return systemState.getClockTime() - op.getJob().getReleaseTime(),1
            elif tree[index].name == 'SLACK' or tree[index].name == 'SL':
                return op.getJob().getDueDate() - systemState.getClockTime() - op.getWorkRemaining(),1
            else:
                return float(tree[index].value), 1
        else:
            print("tree[index].name: ", tree[index].name)



    def protected_div(self, left, right):
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.divide(left, right)
            if isinstance(x, np.ndarray):
                x[np.isinf(x)] = 1
                x[np.isnan(x)] = 1
            elif np.isinf(x) or np.isnan(x):
                x = 1
        return x