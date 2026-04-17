import sys
from tabulate import tabulate
import pandas as pd

import MTGP.LoadIndividual
import GPLS.LoadIndividual
import GSGP.LoadIndividual
import main_experiment_MTGP
import newSimulatorBasedOnJava.RL.sequencingRuleTrain.LoadFile as sequencingLoadFile
import newSimulatorBasedOnJava.RL.routingRuleTrain.LoadFile as routingLoadFile
from wilcoxonTest.wilcoxonTest import doWilcoxonTest

sys.path


if __name__ == '__main__':
    # dataset_name = str(sys.argv[1])
    # mean_of_all_run(dataset_name)

    # all_dataset_name = ['HH', 'HL', 'LH', 'LL']
    # # all_dataset_name = ['HL', 'LH', 'LL']
    # for i in range(len(all_dataset_name)):
    #     print('Result on dataset: ' + all_dataset_name[i])
    #     dataset_name = all_dataset_name[i]
    #     mean_of_all_run(dataset_name)

    # all_dataset_name = ['HH', 'HL', 'LH', 'LL']
    # ruleType = "sequencing"
    ruleType = "routing"
    all_dataset_name = ['total-tardiness_0.95_1.5']
    algos = ['RL']
    runs = 1

    for dataset_index in range(len(all_dataset_name)):
        sum_RL = []
        dataset_name = all_dataset_name[dataset_index]
        print('\nDataset: ' + dataset_name + ': ')
        for run in range(runs):
            for i in range(len(algos)):
                if algos[i] == 'RL' and ruleType == "sequencing":
                    training_loss = sequencingLoadFile.load_training_loss(run, all_dataset_name[dataset_index])
                    if run == 0:
                        sum_RL = training_loss.T
                    else:
                        sum_RL = sum_RL + training_loss.T
                if algos[i] == 'RL' and ruleType == "routing":
                    training_loss = routingLoadFile.load_training_loss(run, all_dataset_name[dataset_index])
                    if run == 0:
                        sum_RL = training_loss.T
                    else:
                        sum_RL = sum_RL + training_loss.T

        sum_dict = {'RL': sum_RL / runs}
        data = pd.DataFrame.from_dict(sum_dict)
        if ruleType == "sequencing":
            addressFinal = './sequencing_models/scenario_' + str(dataset_name) + '/mean_training_loss.xlsx'
            data.to_excel(addressFinal, index=False)
        if ruleType == "routing":
            addressFinal = './routing_models/scenario_' + str(dataset_name) + '/mean_training_loss.xlsx'
            data.to_excel(addressFinal, index=False)

