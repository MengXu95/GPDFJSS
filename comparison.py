import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob


def compare_training_logs(log_dir="./training_logs"):
    """Compare training logs from multiple models"""

    # Find all log files
    log_files = glob(os.path.join(log_dir, "*_training_log_*.xlsx"))

    if not log_files:
        print(f"No training logs found in {log_dir}")
        return

    print(f"Found {len(log_files)} training log files\n")

    # Create comparison DataFrame
    comparison_data = []

    for log_file in log_files:
        # Read summary sheet
        summary_df = pd.read_excel(log_file, sheet_name='Summary')
        epoch_df = pd.read_excel(log_file, sheet_name='Epoch Details')

        model_name = os.path.basename(log_file).split('_training_log_')[0]

        comparison_data.append({
            'Model': model_name,
            'Total Time (min)': summary_df[summary_df['Metric'] == 'Total Training Time (minutes)']['Value'].values[0],
            'Avg Epoch Time (s)': summary_df[summary_df['Metric'] == 'Average Epoch Time (seconds)']['Value'].values[0],
            'Final Loss': summary_df[summary_df['Metric'] == 'Final Train Loss']['Value'].values[0],
            'Best Loss': summary_df[summary_df['Metric'] == 'Best Train Loss']['Value'].values[0],
            'Best Epoch': int(summary_df[summary_df['Metric'] == 'Epoch with Best Loss']['Value'].values[0]),
            'Num Epochs': int(summary_df[summary_df['Metric'] == 'Number of Epochs']['Value'].values[0])
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Print comparison
    print("=" * 100)
    print("Training Comparison Summary")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("=" * 100)

    # Save comparison
    comparison_file = os.path.join(log_dir, "model_comparison.xlsx")
    comparison_df.to_excel(comparison_file, index=False)
    print(f"\nComparison saved to: {comparison_file}")

    # Plot comparison
    plot_training_comparison(log_files, log_dir)

    return comparison_df


def plot_training_comparison(log_files, log_dir="./training_logs"):
    """Create comparison plots"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Comparison: LSTM vs Transformer vs CVAE', fontsize=16)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, log_file in enumerate(log_files):
        epoch_df = pd.read_excel(log_file, sheet_name='Epoch Details')
        model_name = os.path.basename(log_file).split('_training_log_')[0]
        color = colors[idx % len(colors)]

        # Plot 1: Training Loss
        axes[0, 0].plot(epoch_df['epoch'], epoch_df['train_loss'],
                        label=model_name, color=color, linewidth=2)

        # Plot 2: Epoch Time
        axes[0, 1].plot(epoch_df['epoch'], epoch_df['epoch_time_seconds'],
                        label=model_name, color=color, linewidth=2)

        # Plot 3: Cumulative Time
        axes[1, 0].plot(epoch_df['epoch'], epoch_df['cumulative_time_seconds'] / 60,
                        label=model_name, color=color, linewidth=2)

        # Plot 4: Loss vs Time
        axes[1, 1].plot(epoch_df['cumulative_time_seconds'] / 60, epoch_df['train_loss'],
                        label=model_name, color=color, linewidth=2)

    # Configure plots
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss per Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].set_title('Time per Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Cumulative Time (minutes)')
    axes[1, 0].set_title('Total Training Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Training Time (minutes)')
    axes[1, 1].set_ylabel('Training Loss')
    axes[1, 1].set_title('Loss vs Training Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(log_dir, "training_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_file}")
    plt.show()


# Usage example
if __name__ == "__main__":
    # After training all models, compare them
    comparison_df = compare_training_logs("./training_logs")
