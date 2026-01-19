"""
Visualization Tools for Federated Learning Data Partitions

Provides functions for visualizing client data distributions:
- Per-client class histograms
- Heatmaps (clients x classes)
- Distribution comparison overlays
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


def plot_client_histograms(client_data_map, labels, num_classes,
                           num_clients_to_show=5, title="Client Class Distributions",
                           save_path=None):
    """
    Plot class distribution histograms for selected clients.

    Args:
        client_data_map: dict mapping client_id -> array of indices
        labels: array of all labels
        num_classes: number of classes
        num_clients_to_show: number of clients to display
        title: plot title
        save_path: path to save figure (optional)
    """
    counts = get_class_counts(client_data_map, labels, num_classes)
    num_clients = min(num_clients_to_show, len(client_data_map))

    fig, axes = plt.subplots(1, num_clients, figsize=(3 * num_clients, 4))
    if num_clients == 1:
        axes = [axes]

    class_labels = list(range(num_classes))

    for i, ax in enumerate(axes):
        ax.bar(class_labels, counts[i], color='steelblue', edgecolor='black')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title(f'Client {i}')
        ax.set_xticks(class_labels)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        ensure_dirs()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved histogram to {save_path}")

    plt.close()
    return fig


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


def plot_distribution_comparison(client_data_map, labels, num_classes,
                                  title="Client vs Global Distribution",
                                  save_path=None):
    """
    Plot overlay of all client distributions vs global distribution.

    Args:
        client_data_map: dict mapping client_id -> array of indices
        labels: array of all labels
        num_classes: number of classes
        title: plot title
        save_path: path to save figure (optional)
    """
    labels = np.array(labels)
    counts = get_class_counts(client_data_map, labels, num_classes)

    # Normalize to distributions
    distributions = counts / counts.sum(axis=1, keepdims=True)

    # Global distribution
    global_counts = np.bincount(labels, minlength=num_classes)
    global_dist = global_counts / global_counts.sum()

    fig, ax = plt.subplots(figsize=(10, 6))

    class_labels = np.arange(num_classes)

    # Plot each client distribution (thin, semi-transparent lines)
    for i in range(len(client_data_map)):
        ax.plot(class_labels, distributions[i], 'b-', alpha=0.3, linewidth=1)

    # Plot global distribution (thick black line)
    ax.plot(class_labels, global_dist, 'k-', linewidth=3, label='Global')

    # Plot mean client distribution (thick red dashed line)
    mean_dist = distributions.mean(axis=0)
    ax.plot(class_labels, mean_dist, 'r--', linewidth=2, label='Mean Client')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(class_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        ensure_dirs()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution comparison to {save_path}")

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

    # Histograms
    plot_client_histograms(
        client_data_map, labels, num_classes,
        num_clients_to_show=5,
        title=f"{dataset_name.upper()} - {method_name} ({param_value}) - Client Histograms",
        save_path=str(FIGURES_DIR / f"{base_name}_histograms.png")
    )

    # Heatmap
    plot_heatmap(
        client_data_map, labels, num_classes,
        title=f"{dataset_name.upper()} - {method_name} ({param_value}) - Heatmap",
        save_path=str(FIGURES_DIR / f"{base_name}_heatmap.png"),
        normalize=True
    )

    # Distribution comparison
    plot_distribution_comparison(
        client_data_map, labels, num_classes,
        title=f"{dataset_name.upper()} - {method_name} ({param_value}) - Distribution Comparison",
        save_path=str(FIGURES_DIR / f"{base_name}_distribution.png")
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
