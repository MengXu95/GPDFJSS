import random
import time

from deap import tools
import numpy as np
from TransGP.archive_gp import saveFile
from MTGPforNewSimulatorBasedOnJava.selection import selElitistAndTournament


def varAnd(population, toolbox, cxpb, mutpb, reppb):
    offspring = [toolbox.clone(ind) for ind in population]
    total = cxpb + mutpb + reppb
    crossover_limit = cxpb / total
    mutation_limit = (cxpb + mutpb) / total
    i = 1
    while i < len(offspring):
        random_value = random.random()
        if random_value < crossover_limit:
            if (offspring[i - 1] == offspring[i]) :
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        elif random_value < mutation_limit:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
        else:
            del offspring[i].fitness.values
            i = i + 1
    return offspring

#todo: need to check if this is right
def sortPopulation(toolbox, population):
    populationCopy = [toolbox.clone(ind) for ind in population]
    popsize = len(population)

    for j in range(popsize):
        sign = False
        for i in range(popsize-1-j):
            sum_fit_i = np.sum(populationCopy[i].fitness.values)
            sum_fit_i_1 = np.sum(populationCopy[i+1].fitness.values)
            if sum_fit_i > sum_fit_i_1:
                populationCopy[i], populationCopy[i+1] = populationCopy[i+1], populationCopy[i]
                sign = True
        if not sign:
            break

    #FOR CHECK
    # pop_fit = [np.sum(ind.fitness.values) for ind in
    #            populationCopy]
    # print(pop_fit)
    return populationCopy


def eaSimple(population, toolbox, cxpb, mutpb, reppb, elitism, ngen, seedRotate, rd, stats=None,halloffame=None, verbose=__debug__, seed = __debug__, dataset_name=__debug__, wall_clock_limit_seconds=None, start_time=None):
    # initialise the random seed of each generation
    # randomSeed_ngen = []
    # for i in range((ngen + 1)):
    #     randomSeed_ngen.append(np.random.randint(2000000000))

    # Save top 10 individuals from each generation for diffusion model training
    # 20205.4.17 Xu Meng
    archive = []
    if start_time is None:
        start_time = time.time()

    def budget_exhausted(next_gen):
        if wall_clock_limit_seconds is None:
            return False
        elapsed = time.time() - start_time
        if elapsed >= wall_clock_limit_seconds:
            print(
                "Wall-clock budget reached before generation " + str(next_gen) +
                ": elapsed=" + str(elapsed) +
                " seconds, budget=" + str(wall_clock_limit_seconds) + " seconds"
            )
            return True
        return False

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    min_fitness = []
    best_ind_all_gen = [] #add by mengxu
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # rd['seed'] = randomSeed_ngen[0]
    timeBeforeEva = time.time()
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    timeAfterEva = time.time()
    print("Duration of evaluation: " + str(timeAfterEva - timeBeforeEva))
        
    pop_fit = [ind.fitness.values[0] for ind in population]
    min_fitness.append(min(pop_fit))
    # add by mengxu 2022.10.26
    best_index = np.argmin(pop_fit)
    best_ind_all_gen.append(population[best_index])  # add by mengxu
    p_one = population[best_index]
    saveFile.save_individual_each_gen_to_txt(seed, dataset_name, p_one, 0)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    if budget_exhausted(1):
        saveFile.save_each_gen_top_archive_meng(seed, dataset_name, archive)
        return population, logbook, min_fitness, best_ind_all_gen

    # Begin the generational process
    for gen in range(1, ngen + 1):
        if budget_exhausted(gen):
            break

        # Added by mengxu to do seed rotation
        if seedRotate:
            rd['shop_floor'].rotateSeed()
        # Select the next generation individuals
        sorted_pop = sortPopulation(toolbox, population)
        sorted_elite = sorted_pop[:elitism]  # modified by mengxu 2022.10.29
        # sorted_elite = sorted(population, key=attrgetter("fitness"), reverse=True)[:elitism]
        if gen > 30:
            num = 0
            index = 0
            save_fitness = []
            save_archive = []
            while num < elitism*2:
                ind = sorted_pop[index]
                ind_fit = ind.fitness.values[0]
                if ind_fit not in save_fitness:
                    save_fitness.append(ind_fit)
                    save_archive.append(ind)
                    num = num + 1
                index = index + 1

            # save_archive = sorted_pop[:elitism*2] # original
            archive = archive + save_archive # added by mengxu to save top individuals from every generation

        offspring = toolbox.select(population, len(population)-elitism)

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb, reppb)

        # Evaluate the sorted_elite with an invalid fitness as we rotate seed, add by mengxu
        # invalid_elite_ind = [ind for ind in sorted_elite if not ind.fitness.valid]
        invalid_elite_ind = sorted_elite #modified by mengxu, as we rotate seed, no matter it is valid or not valid, we need to re-evaluate
        for ind in invalid_elite_ind:
            del ind.fitness.values
        fitnesses_elite = toolbox.map(toolbox.evaluate, invalid_elite_ind)
        for ind, fit in zip(invalid_elite_ind, fitnesses_elite):
            ind.fitness.values = fit

        # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        invalid_ind = offspring #modified by mengxu, as we rotate seed, no matter it is valid or not valid, we need to re-evaluate
        for ind in invalid_ind:
            del ind.fitness.values
        timeBeforeEva = time.time()
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        timeAfterEva = time.time()
        print("Duration of evaluation: " + str(timeAfterEva - timeBeforeEva))

        # Replace the current population by the offspring
        population[:] = invalid_elite_ind + invalid_ind

        # modified by mengxu
        if halloffame is not None:
            halloffame.clear()  # add by mengxu
            halloffame.update(population)

        # add by mengxu 2022.10.26
        pop_fit = [ind.fitness.values[0] for ind in population]
        best_index = np.argmin(pop_fit)
        best_ind_all_gen.append(population[best_index])  # add by mengxu
        p_one = population[best_index]
        saveFile.save_individual_each_gen_to_txt(seed, dataset_name, p_one, gen)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        pop_fit = [ind.fitness.values[0] for ind in population]  ######selection from author
        min_fitness.append(min(pop_fit))
    saveFile.save_each_gen_top_archive_meng(seed, dataset_name, archive)
    return population, logbook, min_fitness, best_ind_all_gen
