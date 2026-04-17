from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import DiffusionModel.diffusionModel_sequencing as diffusionModel

class EvolutionVisualization:
    def __init__(self):
        self.history = []

    def dimentionalPCAReduction(self, population):
        vec_a, vec_b = [], []
        for ind in population:
            ind_a_prefix_tokens = diffusionModel.primitive_tree_to_string_tree(ind[0])
            tree_a = diffusionModel.prefix_list_to_tree(ind_a_prefix_tokens)
            vec_a.append(diffusionModel.encode_tree_to_vector(tree_a))

            ind_b_prefix_tokens = diffusionModel.primitive_tree_to_string_tree(ind[1])
            tree_b = diffusionModel.prefix_list_to_tree(ind_b_prefix_tokens)
            vec_b.append(diffusionModel.encode_tree_to_vector(tree_b))

        pca_a = PCA(n_components=1).fit_transform(vec_a)
        pca_b = PCA(n_components=1).fit_transform(vec_b)
        return np.hstack([pca_a, pca_b])

    def record_generation(self, population, fitness, tracked_index):
        points = self.dimentionalPCAReduction(population)
        best_idx = np.argmin(fitness)
        self.history.append((points, best_idx, tracked_index))

    def plot_animated_generations(self, save_path="evolution_animation.html"):
        frames = []
        initial_points, initial_best, initial_tracked = self.history[0]

        for gen_idx, (points, best_idx, tracked_idx) in enumerate(self.history):
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', marker=dict(color='blue'), name='Individuals'),
                    go.Scatter(x=[points[best_idx, 0]], y=[points[best_idx, 1]],
                               mode='markers', marker=dict(color='gold', size=12, line=dict(color='black', width=2)), name='Best'),
                    go.Scatter(x=[points[tracked_idx, 0]], y=[points[tracked_idx, 1]],
                               mode='markers', marker=dict(color='red', symbol='x', size=10), name='Tracked')
                ],
                name=f"Gen {gen_idx}"
            ))

        fig = go.Figure(
            data=[
                go.Scatter(x=initial_points[:, 0], y=initial_points[:, 1], mode='markers', marker=dict(color='blue'), name='Individuals'),
                go.Scatter(x=[initial_points[initial_best, 0]], y=[initial_points[initial_best, 1]],
                           mode='markers', marker=dict(color='gold', size=12, line=dict(color='black', width=2)), name='Best'),
                go.Scatter(x=[initial_points[initial_tracked, 0]], y=[initial_points[initial_tracked, 1]],
                           mode='markers', marker=dict(color='red', symbol='x', size=10), name='Tracked')
            ],
            layout=go.Layout(
                title="Population Evolution Across Generations",
                xaxis=dict(title="Vector A (PCA-1D)"),
                yaxis=dict(title="Vector B (PCA-1D)"),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 1000}, "fromcurrent": True}])]
                )]
            ),
            frames=frames
        )

        fig.write_html(save_path)
        print(f"Animated figure saved to: {save_path}")
        fig.show()

    def plot_static_subplots(self, cols=4):
        total_gens = len(self.history)
        rows = (total_gens + cols - 1) // cols

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Gen {i}" for i in range(total_gens)])

        for idx, (points, best_idx, tracked_idx) in enumerate(self.history):
            row, col = (idx // cols) + 1, (idx % cols) + 1
            fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', marker=dict(color='blue'), showlegend=False), row=row, col=col)
            fig.add_trace(go.Scatter(x=[points[best_idx, 0]], y=[points[best_idx, 1]],
                                     mode='markers', marker=dict(color='gold', size=12, line=dict(color='black', width=2)), showlegend=False), row=row, col=col)
            fig.add_trace(go.Scatter(x=[points[tracked_idx, 0]], y=[points[tracked_idx, 1]],
                                     mode='markers', marker=dict(color='red', symbol='x', size=10), showlegend=False), row=row, col=col)

        fig.update_layout(height=300 * rows, width=300 * cols, title_text="All Generations (Static)")
        fig.show()
