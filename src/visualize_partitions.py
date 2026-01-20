"""
Visualization Tools for Federated Learning Data Partitions

Provides functions for visualizing client data distributions:
- Stacked horizontal bar charts (class proportions per client)
- Heatmaps (clients x classes)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project root is the parent of the src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"


def ensure_dirs():
    """Create output directories if they don't exist."""
    FIGURES_DIR.mkdir(exist_ok=True)


def get_class_counts(client_data_map, labels, num_classes):
    """
    Get class counts for each client.

    Args:
        client_data_map: dict mapping client_id -> array of indices
        labels: array of all labels
        num_classes: number of classes

    Returns:
        numpy array of shape (num_clients, num_classes)
    """
    labels = np.array(labels)
    num_clients = len(client_data_map)
    counts = np.zeros((num_clients, num_classes), dtype=int)

    for client_id, indices in client_data_map.items():
        client_labels = labels[indices]
        for c in range(num_classes):
            counts[client_id, c] = np.sum(client_labels == c)

    return counts


def plot_heatmap(client_data_map, labels, num_classes,
                 title="Client-Class Distribution Heatmap",
                 save_path=None, normalize=False):
    """
    Plot heatmap of class distributions across clients.

    Args:
        client_data_map: dict mapping client_id -> array of indices
        labels: array of all labels
        num_classes: number of classes
        title: plot title
        save_path: path to save figure (optional)
        normalize: if True, normalize each client's distribution to sum to 1
    """
    counts = get_class_counts(client_data_map, labels, num_classes)

    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        counts = counts / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(counts, aspect='auto', cmap='YlOrRd')

    # Labels
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Client', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Ticks
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(len(client_data_map)))

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion' if normalize else 'Count')

    plt.tight_layout()

    if save_path:
        ensure_dirs()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")

    plt.close()
    return fig


def plot_class_distribution_stacked(client_data_map, labels, num_classes,
                                     title="Class Distribution per Client",
                                     save_path=None):
    """
    Plot stacked horizontal bar chart of class proportions for each client.

    Each client is a row, and classes are shown as colored segments
    that sum to 1.0, making it easy to compare distributions.

    Args:
        client_data_map: dict mapping client_id -> array of indices
        labels: array of all labels
        num_classes: number of classes
        title: plot title
        save_path: path to save figure (optional)
    """
    counts = get_class_counts(client_data_map, labels, num_classes)
    num_clients = len(client_data_map)

    # Normalize to proportions (each row sums to 1)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    proportions = counts / row_sums

    # Transpose so each row is a class: shape (num_classes, num_clients)
    proportions = proportions.T

    fig, ax = plt.subplots(figsize=(10, max(6, num_clients * 0.4)))

    # Use a colormap with distinct colors for each class
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    # Create stacked horizontal bars
    client_indices = np.arange(num_clients)
    left = np.zeros(num_clients)

    for class_idx in range(num_classes):
        ax.barh(client_indices, proportions[class_idx], left=left,
                color=colors[class_idx], label=f'Class {class_idx}', height=0.8)
        left += proportions[class_idx]

    ax.set_xlabel('Proportion', fontsize=12)
    ax.set_ylabel('Client', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks(client_indices)
    ax.set_yticklabels([f'{i}' for i in range(num_clients)])
    ax.set_xlim(0, 1)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='Class')

    plt.tight_layout()

    if save_path:
        ensure_dirs()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved stacked distribution to {save_path}")

    plt.close()
    return fig


def plot_sample_count_distribution(client_data_map, title="Samples per Client",
                                    save_path=None):
    """
    Plot bar chart of sample counts per client.

    Args:
        client_data_map: dict mapping client_id -> array of indices
        title: plot title
        save_path: path to save figure (optional)
    """
    sample_counts = [len(indices) for indices in client_data_map.values()]
    client_ids = list(client_data_map.keys())

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(client_ids, sample_counts, color='steelblue', edgecolor='black')
    ax.axhline(y=np.mean(sample_counts), color='red', linestyle='--',
               label=f'Mean: {np.mean(sample_counts):.1f}')

    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()

    plt.tight_layout()

    if save_path:
        ensure_dirs()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sample count distribution to {save_path}")

    plt.close()
    return fig


def visualize_partition(client_data_map, labels, num_classes,
                         method_name, param_value, dataset_name="dataset"):
    """
    Generate all visualizations for a partition.

    Args:
        client_data_map: dict mapping client_id -> array of indices
        labels: array of all labels
        num_classes: number of classes
        method_name: name of partitioning method (e.g., "dirichlet", "sharding")
        param_value: parameter value (e.g., alpha=0.5 or shards=2)
        dataset_name: name of dataset (e.g., "mnist", "cifar10")
    """
    ensure_dirs()

    base_name = f"{dataset_name}_{method_name}_{param_value}"

    print(f"\nGenerating visualizations for {base_name}...")

    # Stacked class distribution (main visualization)
    plot_class_distribution_stacked(
        client_data_map, labels, num_classes,
        title=f"{dataset_name.upper()} - {method_name} ({param_value}) - Class Distribution",
        save_path=str(FIGURES_DIR / f"{base_name}_class_distribution.png")
    )

    # Heatmap
    plot_heatmap(
        client_data_map, labels, num_classes,
        title=f"{dataset_name.upper()} - {method_name} ({param_value}) - Heatmap",
        save_path=str(FIGURES_DIR / f"{base_name}_heatmap.png"),
        normalize=True
    )

    # Sample counts
    plot_sample_count_distribution(
        client_data_map,
        title=f"{dataset_name.upper()} - {method_name} ({param_value}) - Samples per Client",
        save_path=str(FIGURES_DIR / f"{base_name}_samples.png")
    )


def load_partition(filepath):
    """
    Load a partition from NPZ file.

    Args:
        filepath: path to NPZ file

    Returns:
        dict mapping client_id -> array of indices
    """
    data = np.load(filepath)
    client_data_map = {}
    for key in data.files:
        client_id = int(key.split('_')[1])
        client_data_map[client_id] = data[key]
    return client_data_map


def main():
    """Example usage: visualize a sample partition."""
    import dataset

    print("=" * 60)
    print("Partition Visualization Example")
    print("=" * 60)

    ensure_dirs()

    # Load MNIST
    print("\nLoading MNIST dataset...")
    train_dataset, _ = dataset.load_mnist()
    labels = np.array(train_dataset.targets)

    # Create example partitions
    num_classes = 10

    # Dirichlet with alpha=0.5
    print("\nGenerating Dirichlet (alpha=0.5) partition...")
    client_data_map = dataset.partition_dirichlet(labels, num_clients=10, alpha=0.5, seed=42)
    visualize_partition(client_data_map, labels, num_classes,
                         "dirichlet", "alpha0.5", "mnist")

    # Sharding with 2 shards per client
    print("\nGenerating Sharding (2 shards) partition...")
    client_data_map = dataset.partition_sharding(labels, num_clients=10, shards_per_client=2, seed=42)
    visualize_partition(client_data_map, labels, num_classes,
                         "sharding", "shards2", "mnist")

    print("\n" + "=" * 60)
    print(f"Visualizations saved to {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
