import pickle5 as pickle
import numpy as np


def load_training_time(randomSeeds, dataSetName): # save individual as txt by mengxu
    folder = './MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_running_time' + dataSetName + '.npy'
    training_time = np.load(folder)

    return training_time

def load_training_loss(run_seed, dataset_name): # save individual as txt by mengxu
    folder = './routing_models/scenario_' + str(dataset_name) + '/' + str(run_seed) + '_training_loss' + '.npy'
    training_loss = np.load(folder)

    return training_loss
