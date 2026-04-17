

class Objective():
    MAKESPAN = "makespan"
    MEAN_FLOWTIME = "mean-flowtime"
    MAX_FLOWTIME = "max-flowtime"
    MEAN_TARDINESS = "mean-tardiness"
    TOTAL_TARDINESS = "total-tardiness"
    MAX_TARDINESS = "max-tardiness"
    MEAN_WEIGHTED_FLOWTIME = "mean-weighted-flowtime"
    MAX_WEIGHTED_FLOWTIME = "max-weighted-flowtime"
    MEAN_WEIGHTED_TARDINESS = "mean-weighted-tardiness"
    MAX_WEIGHTED_TARDINESS = "max-weighted-tardiness"

    def __init__(self, name, **kwargs):
        self.name = name

    def getName(self):
        return self.name


