import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import wilcoxon

# Configuration
algorithms = ['baseDiffGP', 'TransGP']
dataset_name = 'Fmean-0.95-1.5'  # replace with your dataset name

type = 'test_fitness'
y_title = 'Test Performance'

# type = 'training_time'
# y_title = 'Training Time'

# type = 'train_fitness'
# y_title = 'Training Performance'

# type = 'individual_size'
# y_title = 'Individual Size'

num_runs = 30

# Store all results: {algo: [fitness arrays]}
results_by_algo = {}

for algo in algorithms:
    fitness_runs = []

    for seed in range(num_runs):
        filepath = f"C:/Users/I3Nexus/Desktop/2025 Work/2025 paper/Paper4-TransformerGuidedMutationGP/Code related/{algo}/test/scenario_{dataset_name}/{seed}_{dataset_name}_test.csv"
        if not os.path.exists(filepath):
            print(f"Missing file: {filepath}")
            continue

        df = pd.read_csv(filepath)
        # Assume train_fitness column contains fitness array as a string of list
        # Example: "[1.0, 0.9, 0.85, ...]"
        fitness_str = df[type]
        fitness = np.array(fitness_str)  # convert string to list
        fitness_runs.append(fitness)

    results_by_algo[algo] = np.array(fitness_runs)

# Compute statistics and plot
fig = go.Figure()

for algo, runs in results_by_algo.items():
    if runs.shape[0] == 0:
        continue
    mean_fitness = runs.mean(axis=0)
    std_fitness = runs.std(axis=0)
    generations = list(range(len(mean_fitness)))

    # Mean curve
    fig.add_trace(go.Scatter(
        x=generations,
        y=mean_fitness,
        mode='lines',
        name=f'{algo} Mean',
        line=dict(width=2)
    ))

    # Shaded area for std
    fig.add_trace(go.Scatter(
        x=generations + generations[::-1],
        y=(mean_fitness + std_fitness).tolist() + (mean_fitness - std_fitness)[::-1].tolist(),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name=f'{algo} Std'
    ))


fig.update_layout(
    # title='Convergence Curve with Mean and Std Dev',
    xaxis_title='Generation',
    yaxis_title=y_title,
    template='plotly_white'
)

fig.show()


# Wilcoxon test function
def perform_wilcoxon_test(results_by_algo, algo1, algo2, alpha=0.05):
    """ Perform Wilcoxon signed-rank test on the final generation results """
    runs1 = results_by_algo[algo1]
    runs2 = results_by_algo[algo2]

    if runs1.shape[0] != runs2.shape[0]:
        raise ValueError("Both algorithms must have the same number of runs")

    final_gen1 = runs1[:, -1]  # Last generation values
    final_gen2 = runs2[:, -1]

    stat, p_value = wilcoxon(final_gen1, final_gen2)
    print(f"\nWilcoxon Test between {algo1} and {algo2}:")
    print(f"Statistic = {stat:.4f}, p-value = {p_value:.4f}")
    if p_value < alpha:
        if np.mean(final_gen1) < np.mean(final_gen2):
            print(f"Result: {algo1} is significant better than {algo2}")
        else:
            print(f"Result: {algo2} is significant better than {algo1}")
        # print(f"Result: Significant difference (p < {alpha})")
    else:
        print(f"Result: No significant difference (p >= {alpha})")


# Example usage
perform_wilcoxon_test(results_by_algo, 'baseDiffGP', 'TransGP', alpha=0.05)
