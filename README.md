# GPDFJSS

This repository contains our work on dynamic flexible job shop scheduling in Python. The main entry of the project is [main.py](main.py), which is used to launch the core experiments and testing workflow.

## Project overview

The folder [newSimulatorBasedOnJava](newSimulatorBasedOnJava) contains a Python implementation of the DFJSS simulator that we developed based on the Java simulator from https://github.com/meiyi1986/GPJSS. We hope this version is useful for Python users and researchers.

On top of this simulator, we implement a multi-tree genetic programming approach for dispatching rule evolution. The MTGP method is built with the DEAP package and is mainly organized in [MTGPforNewSimulatorBasedOnJava](MTGPforNewSimulatorBasedOnJava).

## Entry point

The main entry of all runnable code is [main.py](main.py). This script:

1. Sets the project root.
2. Reads command-line arguments for the dataset, objective, utilization level, random seed, and algorithm name.
3. Dispatches execution to the selected algorithm.

## Structure of main.py

The workflow in [main.py](main.py) is simple and clear:

- Input arguments:
  - dataset name
  - scheduling objective
  - utilization level
  - seed
  - algorithm selector
- Root setup:
  - ensures the working directory is the project root
- Algorithm dispatch:
  - chooses which experiment pipeline to run

## Algorithms currently supported in main.py

- MTGP  
  Runs the main multi-tree genetic programming training/evolution process on the Python DFJSS simulator.

- MTGP-test  
  Runs testing and result-saving for evolved MTGP rules on the target scheduling scenarios.

## Main folders

- [main.py](main.py): overall experiment entry point
- [newSimulatorBasedOnJava](newSimulatorBasedOnJava): Python DFJSS simulator
- [MTGPforNewSimulatorBasedOnJava](MTGPforNewSimulatorBasedOnJava): DEAP-based MTGP implementation
- [Test](Test): evaluation and result saving
- [Transformer](Transformer): transformer-based models and checkpoints
- [paintingAnalysis](paintingAnalysis): plotting and convergence analysis
- [Utils](Utils): utility scripts for rule analysis and visualization
- [wilcoxonTest](wilcoxonTest): statistical comparison tools

## Example usage

Run the main program with:

python main.py DATASET OBJECTIVE UTILIZATION SEED ALGO

For example:

python main.py Fmax-0.85-1.5 max-flowtime 1.5 0 MTGP

