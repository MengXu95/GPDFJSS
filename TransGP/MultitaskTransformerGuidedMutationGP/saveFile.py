import pickle
import numpy as np
import os


def scenario_dir(dataSetName):
    train_root = os.environ.get('TRANSGP_TRAIN_ROOT', './MultitaskTransformerGuidedMutationGP/train')
    path = os.path.join(train_root, 'scenario_' + str(dataSetName)) + '/'
    os.makedirs(path, exist_ok=True)
    return path


def save_individual(randomSeeds, dataSetName,individuals):
    with open(scenario_dir(dataSetName) + str(randomSeeds) + '_' + dataSetName+'.pickle', 'wb') as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return

def save_each_gen_best_individual_meng(randomSeeds, dataSetName, best_ind_all_gen):
    individual_dict = {}

    for gen in range(len(best_ind_all_gen)):
        best_ind = best_ind_all_gen[gen]

        sequencing = best_ind[0]
        routing = best_ind[1]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)

        routing_list = []
        for i in range(len(routing)):
            routing_list.append(routing[i].name)

        individual.append(sequencing_list)
        individual.append(routing_list)

        individual_dict.__setitem__(gen, individual)

    # fileName_individual = open('./MTGP/train/' + str(randomSeeds) + '_meng_individual_' + dataSetName + '.pkl', "wb")
    # pickle.dump(individual_dict, fileName_individual)
    with open(scenario_dir(dataSetName) + str(randomSeeds) + '_meng_individual_' + dataSetName + '.pkl', "wb") as fileName_individual:
        pickle.dump(individual_dict , fileName_individual)

    return

def save_each_gen_top_archive_meng(randomSeeds, dataSetName, best_ind_all_gen):
    individual_dict = {}

    for gen in range(len(best_ind_all_gen)):
        best_ind = best_ind_all_gen[gen]

        sequencing = best_ind[0]
        routing = best_ind[1]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)

        routing_list = []
        for i in range(len(routing)):
            routing_list.append(routing[i].name)

        individual.append(sequencing_list)
        individual.append(routing_list)

        individual_dict.__setitem__(gen, individual)

    # fileName_individual = open('./MTGP/train/' + str(randomSeeds) + '_meng_individual_' + dataSetName + '.pkl', "wb")
    # pickle.dump(individual_dict, fileName_individual)
    with open(scenario_dir(dataSetName) + str(randomSeeds) + '_meng_archive_' + dataSetName + '.pkl', "wb") as fileName_individual:
        pickle.dump(individual_dict , fileName_individual)

    return


def save_top_inds_final_gen_meng(randomSeeds, dataSetName, top_inds_final_gen):
    individual_dict = {}

    for idx in range(len(top_inds_final_gen)):
        best_ind = top_inds_final_gen[idx]

        sequencing = best_ind[0]
        routing = best_ind[1]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)

        routing_list = []
        for i in range(len(routing)):
            routing_list.append(routing[i].name)

        individual.append(sequencing_list)
        individual.append(routing_list)

        individual_dict.__setitem__(idx, individual)

    with open(scenario_dir(dataSetName) + str(randomSeeds) + '_meng_top_individuals_final_gen_' + dataSetName + '.pkl', "wb") as fileName_individual:
        pickle.dump(individual_dict, fileName_individual)

    return

def save_individual_to_txt(randomSeeds, dataSetName,individuals): # save individual as txt by mengxu
    file = open(scenario_dir(dataSetName) + str(randomSeeds) + '_' + dataSetName+'.txt', 'w')
    file.write('Individual:\n')
    file.write('Tree 0:\n') #routing rule
    file.write(str(individuals[0]) + '\n')
    file.write('Tree 1:\n') #sequencing rule
    file.write(str(individuals[1]) + '\n')

    file.close()
    return

def clear_individual_each_gen_to_txt(randomSeeds, dataSetName): # save individual as txt by mengxu
    file = open(scenario_dir(dataSetName) + str(randomSeeds) + '_' + dataSetName+'_each_gen.txt', 'w') # 'w' represent coverage, 'a' denotes not coverage
    file.write("Best individuals from each gen:\n")
    file.close()
    return

def save_individual_each_gen_to_txt(randomSeeds, dataSetName, individuals, gen): # save individual as txt by mengxu
    file = open(scenario_dir(dataSetName) + str(randomSeeds) + '_' + dataSetName+'_each_gen.txt', 'a') # 'w' represent coverage, 'a' denotes not coverage
    file.write('\nGen: ' + str(gen) + '\n')
    file.write('Individual:\n')
    file.write('Tree 0:\n') #routing rule
    file.write(str(individuals[0]) + '\n')
    file.write('Tree 1:\n') #sequencing rule
    file.write(str(individuals[1]) + '\n')

    file.close()
    return


def save_archive(randomSeeds, dataSetName,individuals):
    with open(scenario_dir(dataSetName) + str(randomSeeds) + '_archive' + dataSetName+'.pickle', 'wb') as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return

def save_pop(randomSeeds, dataSetName,individuals):
    with open(scenario_dir(dataSetName) + str(randomSeeds) + '_pop' + dataSetName+'.pickle', 'wb') as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return


def saveMinFitness(randomSeeds, dataSetName, min_fitness):
    fileName1= scenario_dir(dataSetName) + str(randomSeeds)+'_min_fitness' + dataSetName
    np.save(fileName1, min_fitness)
    return

def saveRunningTime(randomSeeds, dataSetName, running_time):
    fileName1= scenario_dir(dataSetName) + str(randomSeeds)+'_running_time' + dataSetName
    np.save(fileName1, running_time)
    return
