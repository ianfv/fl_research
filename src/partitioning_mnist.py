"""
MNIST Partitioning Analysis Script

Runs sharding and Dirichlet partitioning with varying parameters,
computes metrics, and saves results.
"""

import json
import numpy as np
from pathlib import Path

import dataset
import metrics

# Project root is the parent of the src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARTITIONS_DIR = PROJECT_ROOT / "partitions"
RESULTS_DIR = PROJECT_ROOT / "results"

# Configuration
NUM_CLIENTS = 10
NUM_CLASSES = 10

# Parameter ranges for analysis
ALPHA_VALUES = [0.1, 0.3, 0.5, 1.0, 5.0, 10.0]
SHARDS_PER_CLIENT_VALUES = [1, 2, 5, 10]


def ensure_dirs():
    """Create output directories if they don't exist."""
    PARTITIONS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def save_partition(client_data_map, filename):
    """Save partition indices to NPZ file."""
    filepath = PARTITIONS_DIR / filename
    np.savez_compressed(
        str(filepath),
        **{f'client_{i}': indices for i, indices in client_data_map.items()}
    )
    print(f"  Saved partition to {filepath}")


def save_metrics(metrics_dict, filename):
    """Save metrics to JSON file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Saved metrics to {filepath}")


def run_dirichlet_analysis(labels, num_clients, alpha_values):
    """
    Run Dirichlet partitioning with varying alpha values.

    Args:
        labels: numpy array of labels
        num_clients: number of clients
        alpha_values: list of alpha values to test

    Returns:
        dict mapping alpha -> metrics
    """
    print("\n" + "=" * 60)
    print("Dirichlet Partitioning Analysis")
    print("=" * 60)

    results = {}

    for alpha in alpha_values:
        print(f"\nAlpha = {alpha}:")

        # Partition data
        client_data_map = dataset.partition_dirichlet(
            labels, num_clients=num_clients, alpha=alpha, seed=42
        )

        # Compute metrics
        partition_metrics = metrics.compute_all_metrics(
            client_data_map, labels, NUM_CLASSES
        )

        # Add partition parameters
        partition_metrics['parameters'] = {
            'method': 'dirichlet',
            'alpha': alpha,
            'num_clients': num_clients,
            'seed': 42
        }

        results[alpha] = partition_metrics

        # Print summary
        agg = partition_metrics['aggregate']
        print(f"  Entropy: {agg['entropy']['mean']:.4f} +/- {agg['entropy']['std']:.4f}")
        print(f"  KL Div:  {agg['kl_divergence']['mean']:.4f} +/- {agg['kl_divergence']['std']:.4f}")
        print(f"  Dom Cls: {agg['dominant_classes']['mean']:.2f}")

        # Save partition and metrics
        save_partition(client_data_map, f"mnist_dirichlet_alpha{alpha}.npz")
        save_metrics(partition_metrics, f"mnist_dirichlet_alpha{alpha}_metrics.json")

    return results


def run_sharding_analysis(labels, num_clients, shards_per_client_values):
    """
    Run sharding partitioning with varying shards_per_client values.

    Args:
        labels: numpy array of labels
        num_clients: number of clients
        shards_per_client_values: list of shards_per_client values to test

    Returns:
        dict mapping shards_per_client -> metrics
    """
    print("\n" + "=" * 60)
    print("Sharding Partitioning Analysis")
    print("=" * 60)

    results = {}

    for shards in shards_per_client_values:
        print(f"\nShards per client = {shards}:")

        # Partition data
        client_data_map = dataset.partition_sharding(
            labels, num_clients=num_clients, shards_per_client=shards, seed=42
        )

        # Compute metrics
        partition_metrics = metrics.compute_all_metrics(
            client_data_map, labels, NUM_CLASSES
        )

        # Add partition parameters
        partition_metrics['parameters'] = {
            'method': 'sharding',
            'shards_per_client': shards,
            'num_clients': num_clients,
            'seed': 42
        }

        results[shards] = partition_metrics

        # Print summary
        agg = partition_metrics['aggregate']
        print(f"  Entropy: {agg['entropy']['mean']:.4f} +/- {agg['entropy']['std']:.4f}")
        print(f"  KL Div:  {agg['kl_divergence']['mean']:.4f} +/- {agg['kl_divergence']['std']:.4f}")
        print(f"  Dom Cls: {agg['dominant_classes']['mean']:.2f}")

        # Save partition and metrics
        save_partition(client_data_map, f"mnist_sharding_shards{shards}.npz")
        save_metrics(partition_metrics, f"mnist_sharding_shards{shards}_metrics.json")

    return results


def main():
    """Main entry point for MNIST partitioning analysis."""
    print("=" * 60)
    print("MNIST Non-IID Partitioning Analysis")
    print("=" * 60)

    # Ensure output directories exist
    ensure_dirs()

    # Load MNIST
    print("\nLoading MNIST dataset...")
    train_dataset, _ = dataset.load_mnist()
    labels = np.array(train_dataset.targets)
    print(f"Loaded {len(labels)} training samples")

    # Run analyses
    dirichlet_results = run_dirichlet_analysis(labels, NUM_CLIENTS, ALPHA_VALUES)
    sharding_results = run_sharding_analysis(labels, NUM_CLIENTS, SHARDS_PER_CLIENT_VALUES)

    # Save combined results
    combined_results = {
        'dataset': 'mnist',
        'num_clients': NUM_CLIENTS,
        'num_classes': NUM_CLASSES,
        'num_samples': len(labels),
        'dirichlet': {
            'alpha_values': ALPHA_VALUES,
            'summary': {
                str(alpha): {
                    'entropy_mean': res['aggregate']['entropy']['mean'],
                    'entropy_std': res['aggregate']['entropy']['std'],
                    'kl_mean': res['aggregate']['kl_divergence']['mean'],
                    'kl_std': res['aggregate']['kl_divergence']['std'],
                    'dominant_classes_mean': res['aggregate']['dominant_classes']['mean']
                }
                for alpha, res in dirichlet_results.items()
            }
        },
        'sharding': {
            'shards_per_client_values': SHARDS_PER_CLIENT_VALUES,
            'summary': {
                str(shards): {
                    'entropy_mean': res['aggregate']['entropy']['mean'],
                    'entropy_std': res['aggregate']['entropy']['std'],
                    'kl_mean': res['aggregate']['kl_divergence']['mean'],
                    'kl_std': res['aggregate']['kl_divergence']['std'],
                    'dominant_classes_mean': res['aggregate']['dominant_classes']['mean']
                }
                for shards, res in sharding_results.items()
            }
        }
    }

    save_metrics(combined_results, "mnist_analysis_summary.json")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Partitions saved to: {PARTITIONS_DIR}")
    print(f"Metrics saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
