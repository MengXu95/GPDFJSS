
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import sys

# import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams['pdf.fonttype'] = 42

fig = plt.figure()
# gs=GridSpec(nrows=3, ncols=4)
gs=GridSpec(nrows=5, ncols=10)

ruleType = "routing"

if ruleType == "sequencing":
    fileName = "RLsequencingtrainingLossCurve.pdf"
if ruleType == "routing":
    fileName = "RLroutingtrainingLossCurve.pdf"

# all_dataset_name = ['HH', 'HL', 'LH', 'LL']
# algos = ['MTGP', 'GPLS']
all_dataset_name = ['total-tardiness_0.95_1.5']
algos = ['RL']


for i in range(0, len(all_dataset_name)):
    ax = fig.add_subplot(gs[0:4, 0:10])
    # if i<2:
    #     ax = fig.add_subplot(gs[0:2, i])
    # else:
    #     ax = fig.add_subplot(gs[2:4, i-2])

    dataset_name = all_dataset_name[i]
    title = 'On dataset ' + all_dataset_name[i]
    plt.gca().set_title(title, fontweight='bold', size=9)

    if ruleType == "sequencing":
        labels = ['RL-sequencing']
        for algo in range(0, len(algos)):
            address = './sequencing_models/scenario_' + str(dataset_name) + '/mean_training_loss.xlsx'
            results = pd.read_excel(address, engine='openpyxl')
            print(results)

            ax.plot(range(0,len(results[algos[algo]])), list(results[algos[algo]]), linestyle = '-', linewidth= 0.5, marker ='.', markersize = 5)
    if ruleType == "routing":
        labels = ['RL-routing']
        for algo in range(0, len(algos)):
            address = './routing_models/scenario_' + str(dataset_name) + '/mean_training_loss.xlsx'
            results = pd.read_excel(address, engine='openpyxl')
            print(results)

            ax.plot(range(0, len(results[algos[algo]])), list(results[algos[algo]]), linestyle='-', linewidth=0.5,
                    marker='.', markersize=5)

plt.figlegend(labels, loc = 'lower center', ncol = 3, bbox_to_anchor = (0.53, 0.02), labelspacing=0., fontsize = 'small')

fig.text(0.5, 0.12, 'Decision points', ha='center', size = 11, fontweight='bold')
fig.text(0, 0.54, 'Training loss', va='center', rotation='vertical', size = 11, fontweight='bold')

plt.tight_layout()
plt.savefig("../paintingAnalysis/" + fileName)
# plt.savefig("test.png")
plt.show()
