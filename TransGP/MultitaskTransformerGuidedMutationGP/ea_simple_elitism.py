import random
import time

import numpy as np
from deap import tools

import Utils.FitnessDiversity as FitnessDiversity
import Utils.RuleSize as RuleSize
from TransGP.MultitaskTransformerGuidedMutationGP import saveFile
from TransGP.Transformer.Transformer_tree_multitask import TransformerTree


def varAnd(population, toolbox, cxpb, mutpb, reppb, transformer_gp):
    offspring = [toolbox.clone(ind) for ind in population]
    total = cxpb + mutpb + reppb
    crossover_limit = cxpb / total
    mutation_limit = (cxpb + mutpb) / total
    i = 1
    while i < len(offspring):
        random_value = random.random()
        if random_value < crossover_limit:
            if (offspring[i - 1] == offspring[i]) :
                offspring[i - 1], = toolbox.mutate(offspring[i - 1], transformer_gp=transformer_gp)
                offspring[i], = toolbox.mutate(offspring[i], transformer_gp=transformer_gp)
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        elif random_value < mutation_limit:
            # print("Before mutation: ")
            # print("Tree 0: ", str(offspring[i][0]))
            # print("Tree 1: ", str(offspring[i][1]))
            offspring[i], = toolbox.mutate(offspring[i], transformer_gp=transformer_gp)
            # print("After mutation: ")
            # print("Tree 0: ", str(offspring[i][0]))
            # print("Tree 1: ", str(offspring[i][1]))
            del offspring[i].fitness.values
            i = i + 1
        else:
            del offspring[i].fitness.values
            i = i + 1
    return offspring

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

#todo: need to check if this is right
def sortPopulation_multitask(toolbox, population, num_task):
    sorted_pop_multitask = []
    for task_id in range(num_task):
        populationCopy = [toolbox.clone(ind) for ind in population if ind.task_tag==task_id]
        popsize = len(populationCopy)

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

        sorted_pop_multitask.append(populationCopy)

    return sorted_pop_multitask

def eaSimple(population, toolbox, cxpb, mutpb, reppb, elitism, ngen, seedRotate, rd, stats=None,halloffame=None, verbose=__debug__, seed = __debug__, dataset_name=__debug__):
    print("----------Transformer-Multitask----------")
    transformer_sequencing = TransformerTree(seed, dataset_name, rule_type="sequencing", epochs=50, lr=1e-4, max_len=128, task_emb_dim=2)
    transformer_routing = TransformerTree(seed, dataset_name, rule_type="routing", epochs=50, lr=1e-4, max_len=128, task_emb_dim=2)
    sequencing_model_dir = f"./Transformer/{transformer_sequencing.rule_type}_transformer_multitask"
    routing_model_dir = f"./Transformer/{transformer_routing.rule_type}_transformer_multitask"
    transformer_sequencing.load_model(sequencing_model_dir) # need to revise based on requirements
    transformer_routing.load_model(routing_model_dir) # need to revise based on requirements

    transformer_gp = [transformer_sequencing, transformer_routing]

    dataset_name_parts = dataset_name.split('-')
    task_utils = dataset_name_parts[1:-1]
    num_task = len(task_utils)

    diversityCalculator = FitnessDiversity.FitnessDiversityCalculator()
    ruleSizeCalculator = RuleSize.RuleSizeCalculator()


    # 20205.4.17 Xu Meng
    archive = []

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    min_fitness = [[] for _ in range(num_task)]
    best_ind_all_gen = [[] for _ in range(num_task)] #add by mengxu
    top_inds_final_gen = [[] for _ in range(num_task)]
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    # rd['seed'] = randomSeed_ngen[0]
    timeBeforeEva = time.time()
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    timeAfterEva = time.time()
    print("Duration of evaluation: " + str(timeAfterEva - timeBeforeEva))
        
    # pop_fit = [ind.fitness.values[0] for ind in population]
    # min_fitness.append(min(pop_fit))
    # # add by mengxu 2022.10.26
    # best_index = np.argmin(pop_fit)
    # best_ind_all_gen.append(population[best_index])  # add by mengxu
    # p_one = population[best_index]
    # saveFile.save_individual_each_gen_to_txt(seed, dataset_name, p_one, 0)
    sortedPop_multitask = sortPopulation_multitask(toolbox, population, num_task)
    # add by mengxu 2025.6.10
    for id in range(len(sortedPop_multitask)):
        sortedPop_eachtask = sortedPop_multitask[id]
        pop_fit_eachtask = [ind.fitness.values[0] for ind in sortedPop_eachtask]
        best_index_eachtask = np.argmin(pop_fit_eachtask)
        best_ind_all_gen[id].append(sortedPop_eachtask[best_index_eachtask])  # add by mengxu
        p_one_eachtask = sortedPop_eachtask[best_index_eachtask]
        best_ind_size_eachtask = len(p_one_eachtask[0]) + len(p_one_eachtask[1])
        # print(f"Task {id} best individual size: ", best_ind_size_eachtask)
        dataset_name_eachtask = dataset_name_parts[0] + '-' + task_utils[id] + '-' + dataset_name_parts[-1]
        saveFile.save_individual_each_gen_to_txt(seed, dataset_name_eachtask, p_one_eachtask, 0)

        # Append the current generation statistics to the logbook
        record = stats.compile(sortedPop_eachtask) if stats else {}
        logbook.record(gen=0, task=id, nevals=len(sortedPop_eachtask), **record)
        if verbose:
            print(logbook.stream)

        pop_fit_eachtask = [ind.fitness.values[0] for ind in sortedPop_eachtask]  ######selection from author
        min_fitness[id].append(min(pop_fit_eachtask))


    if halloffame is not None:
        halloffame.update(population)


    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Added by mengxu to do seed rotation
        if seedRotate:
            for task in rd['shop_floor']:
                task.rotateSeed()
            # rd['shop_floor'].rotateSeed() # original

        # # Select the next generation individuals for single task
        # sortedPop = sortPopulation(toolbox, population)
        # sortedPop_fit = [ind.fitness.values[0] for ind in sortedPop]
        # best_index = np.argmin(sortedPop_fit)
        # sorted_elite = sortedPop[:elitism]  # modified by mengxu 2022.10.29
        # # sorted_elite = sorted(population, key=attrgetter("fitness"), reverse=True)[:elitism]
        # archive = archive + sorted_elite # added by mengxu to save top individuals from every generation

        # Select the next generation individuals for multiple task
        sortedPop_multitask = sortPopulation_multitask(toolbox, population, num_task)
        sortedPop_fit_multitask = []
        sorted_elite_multitask = []
        for id in range(len(sortedPop_multitask)):
            sortedPop_eachtask = sortedPop_multitask[id]
            sortedPop_fit_eachtask = [ind.fitness.values[0] for ind in sortedPop_eachtask]
            sortedPop_fit_multitask.append(sortedPop_fit_eachtask)
            best_index_eachtask = np.argmin(sortedPop_fit_eachtask)
            sorted_elite_eachtask = sortedPop_eachtask[:elitism]  # modified by mengxu 2022.10.29
            sorted_elite_multitask.append(sorted_elite_eachtask)
            archive = archive + sorted_elite_eachtask  # added by mengxu to save top individuals from every generation

            diversity = diversityCalculator.calculate_diversity(sortedPop_eachtask)
            print(f"Task {id} population diversity: {diversity}")
            average_rule_size = ruleSizeCalculator.calculate_average_rule_size(sortedPop_eachtask)
            print(f"Task {id} average rule size: {average_rule_size}")

        # modified by mengxu
        if halloffame is not None:
            halloffame.clear()  # add by mengxu
            halloffame.update(population)


        # offspring = toolbox.select(population, len(population)-elitism)
        # offspring = population[:len(population)-elitism] # for single task
        offspring = []
        for sortedPop_eachtask in sortedPop_multitask:
            offspring = offspring + toolbox.select(sortedPop_eachtask, len(sortedPop_eachtask) - elitism)


        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb, reppb, transformer_gp)

        # Evaluate the sorted_elite with an invalid fitness as we rotate seed, add by mengxu
        # invalid_elite_ind = [ind for ind in sorted_elite if not ind.fitness.valid]
        invalid_elite_ind = [] #modified by mengxu, as we rotate seed, no matter it is valid or not valid, we need to re-evaluate
        for each_sorted_elite in sorted_elite_multitask:
            invalid_elite_ind = invalid_elite_ind + each_sorted_elite

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
        population[:] = invalid_elite_ind + invalid_ind # original

        # Select the next generation individuals for multiple task
        # sortedPop_multitask = sortPopulation_multitask(toolbox, population, num_task)
        population_multitask = []
        for task_id in range(num_task):
            populationCopy = [toolbox.clone(ind) for ind in population if ind.task_tag == task_id]
            population_multitask.append(populationCopy)

        for id in range(len(population_multitask)):
            population_eachtask = population_multitask[id]
            pop_fit_eachtask = [ind.fitness.values[0] for ind in population_eachtask]
            best_index_eachtask = np.argmin(pop_fit_eachtask)
            best_ind_all_gen[id].append(population_eachtask[best_index_eachtask])  # add by mengxu
            p_one_eachtask = population_eachtask[best_index_eachtask]
            best_ind_size_eachtask = len(p_one_eachtask[0]) + len(p_one_eachtask[1])
            # print(f"Task {id} best individual size: {best_ind_size_eachtask}")
            dataset_name_eachtask = dataset_name_parts[0] + '-' + task_utils[id] + '-' + dataset_name_parts[-1]
            saveFile.save_individual_each_gen_to_txt(seed, dataset_name_eachtask, p_one_eachtask, gen)

            # Append the current generation statistics to the logbook
            record = stats.compile(population_eachtask) if stats else {}
            logbook.record(gen=gen, task=id, nevals=len(population_eachtask), **record)
            if verbose:
                print(logbook.stream)

            pop_fit_eachtask = [ind.fitness.values[0] for ind in population_eachtask]  ######selection from author
            min_fitness[id].append(min(pop_fit_eachtask))


    final_sorted_multitask = sortPopulation_multitask(toolbox, population, num_task)
    for task_id, sorted_pop_eachtask in enumerate(final_sorted_multitask):
        if sorted_pop_eachtask:
            top_inds_final_gen[task_id] = [sorted_pop_eachtask[0]]

    # saveFile.save_each_gen_top_archive_meng(seed, dataset_name, archive)

    return population, logbook, min_fitness, best_ind_all_gen, top_inds_final_gen
