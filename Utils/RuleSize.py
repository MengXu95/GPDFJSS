from collections import Counter
import csv
import os

import numpy as np


class RuleSizeCalculator:
    def __init__(self):
        """
        Initialize the PCDiversityCalculator with a population of individuals.

        :param population: List of individuals, where each individual is a list of PC values.
        """


    def calculate_average_rule_size(self, population):
        """
        Calculate the PC diversity by counting how many individuals have the same PC values.

        :return: A dictionary with the number of occurrences of each unique PC set and
                 the number of unique individuals.
        """
        # Count the occurrences of each unique PC value set
        sequencing_rule_sizes = []
        routing_rule_sizes = []

        for ind in population:
            sequencing_rule = ind[0]
            routing_rule = ind[1]
            sequencing_rule_sizes.append(len(sequencing_rule))
            routing_rule_sizes.append(len(routing_rule))

        average_size = np.average(sequencing_rule_sizes) + np.average(routing_rule_sizes)

        return average_size




