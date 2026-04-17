from collections import Counter
import csv
import os

class FitnessDiversityCalculator:
    def __init__(self):
        """
        Initialize the PCDiversityCalculator with a population of individuals.

        :param population: List of individuals, where each individual is a list of PC values.
        """

    def calculate_diversity(self, population):
        """
        Calculate the PC diversity by counting how many individuals have the same PC values.

        :return: A dictionary with the number of occurrences of each unique PC set and
                 the number of unique individuals.
        """
        # Count the occurrences of each unique fitness value set
        fitness = []
        for ind in population:
            fit = ind.fitness.values
            fitness.append(fit)

        counts = Counter(fitness)

        # Calculate the number of unique individuals
        num_unique_individuals = len(counts)

        # Return the PC diversity
        return num_unique_individuals/len(population)




