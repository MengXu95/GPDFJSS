import conditionalVAE.conditional_VAE_tree_multitask as CVAETrain
import LSTMDecoder.LSTM_Decoder_tree_multitask as LSTMTrain
import Transformer.Transformer_tree_multitask as TransformerTrain
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
    dataset_name = str(sys.argv[1]) # e.g., "Fmax-0.75-0.85-0.95-1.5"
    rule_type = str(sys.argv[2]) # e.g., "routing"
    model_type = str(sys.argv[3]) #e.g., CVAE or Transformer or LSTM

    # Example usage:
    set_project_root()
    print("Current working directory:", os.getcwd())  # add this line
    print(f'Train {model_type} of {rule_type} on scenario {dataset_name}:')

    if model_type == 'Transformer':
        TransformerTrain.main_train(dataset_name, rule_type)
    elif model_type == 'CVAE':
        CVAETrain.main_train(dataset_name, rule_type)
    elif model_type == 'LSTM':
        LSTMTrain.main_train(dataset_name, rule_type)