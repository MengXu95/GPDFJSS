import MTGPforNewSimulatorBasedOnJava.GPFC as GPNewSimulatormain
import Test.testAndResultsSave as testAndResultsSave
import sys

import os
import os.path

sys.path

def set_project_root():
    """Sets the project root directory as the current working directory."""
    current_file = os.path.abspath(__file__) # get the absolute path of the current file
    project_root = os.path.dirname(current_file)
    os.chdir(project_root)
    # project_root = os.path.dirname(os.path.dirname(current_file)) # go up two levels to get the project root
    # os.chdir(project_root)

if __name__ == '__main__':
    dataset_name = str(sys.argv[1]) # e.g., "Fmax-0.85-1.5"
    objective = str(sys.argv[2])    # e.g., "max-flowtime"
    utilLevel = float(sys.argv[3])    # e.g., "1.5"
    seed = int(sys.argv[4])         # e.g., "0"
    algo = str(sys.argv[5])         # e.g., "MTGP"
    machine_num = 10

    # Example usage:
    set_project_root()
    print("Current working directory:", os.getcwd())  # add this line

    if algo == 'MTGP':
        print('----------MTGPforNewSimulatorBasedOnJava----------')
        GPNewSimulatormain.main(dataset_name, seed, objective, utilLevel)
    elif algo == 'MTGP-test':
        print('----------MTGP-test----------')
        testAndResultsSave.main("MTGPforNewSimulatorBasedOnJava", dataset_name, seed, objective, utilLevel, machine_num)

