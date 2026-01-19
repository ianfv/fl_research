"""
Method Comparison Tool for Federated Learning Data Partitioning

Provides:
- Metric vs parameter plots for each method
- Side-by-side method comparisons
- Extensible registry for future partitioning methods
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import dataset
import metrics as metrics_module

# Project root is the parent of the src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_dirs():
    """Create output directories if they don't exist."""
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Extensible Method Registry
# =============================================================================

METHODS = {
    'dirichlet': {
        'param_name': 'alpha',
        'param_values': [0.1, 0.3, 0.5, 1.0, 5.0, 10.0],
        'partition_fn': dataset.partition_dirichlet,
        'param_label': r'$\alpha$',
        'description': 'Dirichlet distribution (lower = more non-IID)',
    },
    'sharding': {
        'param_name': 'shards_per_client',
        'param_values': [1, 2, 5, 10],
        'partition_fn': dataset.partition_sharding,
        'param_label': 'Shards per Client',
        'description': 'Pathological sharding (fewer shards = more non-IID)',
    },
    # STUBS FOR FUTURE METHODS:
    'lognormal': {
        'param_name': 'sigma',
        'param_values': [0.1, 0.5, 1.0, 2.0],
        'partition_fn': None,  # TODO: implement
        'param_label': r'$\sigma$',
        'description': 'Log-normal quantity skew',
    },
    'corruptions': {
        'param_name': 'noise_level',
        'param_values': [0.1, 0.3, 0.5],
        'partition_fn': None,  # TODO: implement
        'param_label': 'Noise Level',
        'description': 'Feature corruption skew',
    },
    'clustering': {
        'param_name': 'num_clusters',
        'param_values': [2, 5, 10],
        'partition_fn': None,  # TODO: implement
        'param_label': 'Number of Clusters',
        'description': 'Clustering-based partitioning',
    },
}


# =============================================================================
# Analysis Functions
# =============================================================================

def run_method_analysis(labels, num_classes, method_name, num_clients=10, seed=42):
    """
    Run analysis for a single partitioning method across all its parameter values.

    Args:
        labels: numpy array of labels
        num_classes: number of classes
        method_name: name of method (key in METHODS dict)
        num_clients: number of clients
        seed: random seed

    Returns:
        dict with results for each parameter value
    """
    method = METHODS[method_name]

    if method['partition_fn'] is None:
        print(f"  Method '{method_name}' not yet implemented, skipping...")
        return None

    param_name = method['param_name']
    param_values = method['param_values']
    partition_fn = method['partition_fn']

    results = {
        'method': method_name,
        'param_name': param_name,
        'param_values': param_values,
        'metrics': {}
    }

    for param_value in param_values:
        # Create partition
        kwargs = {
            'targets': labels,
            'num_clients': num_clients,
            param_name: param_value,
            'seed': seed
        }
        client_data_map = partition_fn(**kwargs)

        # Compute metrics
        partition_metrics = metrics_module.compute_all_metrics(
            client_data_map, labels, num_classes
        )

        results['metrics'][param_value] = {
            'entropy_mean': partition_metrics['aggregate']['entropy']['mean'],
            'entropy_std': partition_metrics['aggregate']['entropy']['std'],
            'kl_mean': partition_metrics['aggregate']['kl_divergence']['mean'],
            'kl_std': partition_metrics['aggregate']['kl_divergence']['std'],
            'dominant_classes_mean': partition_metrics['aggregate']['dominant_classes']['mean'],
            'dominant_classes_std': partition_metrics['aggregate']['dominant_classes']['std'],
            'sample_count_std': partition_metrics['aggregate']['sample_counts']['std'],
        }

    return results


def run_all_methods(labels, num_classes, num_clients=10, seed=42):
    """
    Run analysis for all implemented methods.

    Args:
        labels: numpy array of labels
        num_classes: number of classes
        num_clients: number of clients
        seed: random seed

    Returns:
        dict mapping method_name -> results
    """
    all_results = {}

    for method_name in METHODS:
        print(f"\nAnalyzing {method_name}...")
        results = run_method_analysis(labels, num_classes, method_name, num_clients, seed)
        if results is not None:
            all_results[method_name] = results

    return all_results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_metric_vs_param(results, metric_name, ylabel, title=None, save_path=None):
    """
    Plot a metric vs parameter value for a single method.

    Args:
        results: dict from run_method_analysis
        metric_name: name of metric (e.g., 'entropy_mean')
        ylabel: y-axis label
        title: plot title (optional)
        save_path: path to save figure (optional)
    """
    param_values = results['param_values']
    metrics_data = results['metrics']

    mean_key = f'{metric_name}_mean' if not metric_name.endswith('_mean') else metric_name
    std_key = mean_key.replace('_mean', '_std')

    means = [metrics_data[p][mean_key] for p in param_values]
    stds = [metrics_data[p].get(std_key, 0) for p in param_values]

    method = METHODS[results['method']]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(param_values, means, yerr=stds, marker='o', capsize=5,
                linewidth=2, markersize=8)

    ax.set_xlabel(method['param_label'], fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if title is None:
        title = f"{results['method'].title()}: {ylabel} vs {method['param_label']}"
    ax.set_title(title, fontsize=14)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        ensure_dirs()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()
    return fig


def plot_all_metrics_for_method(results, dataset_name="dataset"):
    """
    Generate all metric plots for a single method.

    Args:
        results: dict from run_method_analysis
        dataset_name: name of dataset (for file naming)
    """
    ensure_dirs()
    method_name = results['method']

    metric_configs = [
        ('entropy', 'Entropy (higher = more IID)', 'entropy'),
        ('kl', 'KL Divergence (lower = more IID)', 'kl'),
        ('dominant_classes', 'Dominant Classes for 80%', 'dominant'),
    ]

    for metric_name, ylabel, short_name in metric_configs:
        save_path = str(
            FIGURES_DIR /
            f"{dataset_name}_{method_name}_{short_name}_vs_param.png"
        )
        plot_metric_vs_param(
            results, metric_name, ylabel,
            title=f"{dataset_name.upper()} - {method_name.title()}: {ylabel}",
            save_path=save_path
        )


def plot_methods_comparison(all_results, metric_name, ylabel, dataset_name="dataset",
                             title=None, save_path=None):
    """
    Compare multiple methods on the same plot.

    Args:
        all_results: dict mapping method_name -> results
        metric_name: name of metric to compare
        ylabel: y-axis label
        dataset_name: name of dataset
        title: plot title (optional)
        save_path: path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    mean_key = f'{metric_name}_mean' if not metric_name.endswith('_mean') else metric_name
    std_key = mean_key.replace('_mean', '_std')

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v']

    for i, (method_name, results) in enumerate(all_results.items()):
        param_values = results['param_values']
        metrics_data = results['metrics']

        means = [metrics_data[p][mean_key] for p in param_values]
        stds = [metrics_data[p].get(std_key, 0) for p in param_values]

        method = METHODS[method_name]

        # Normalize x-axis to [0, 1] for comparison
        x_norm = np.linspace(0, 1, len(param_values))

        ax.errorbar(x_norm, means, yerr=stds, marker=markers[i % len(markers)],
                    capsize=5, linewidth=2, markersize=8, color=colors[i],
                    label=f"{method_name.title()} ({method['param_label']})")

    ax.set_xlabel('Parameter (normalized: 0=most non-IID, 1=most IID)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if title is None:
        title = f"{dataset_name.upper()}: Method Comparison - {ylabel}"
    ax.set_title(title, fontsize=14)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        ensure_dirs()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    plt.close()
    return fig


def plot_all_comparisons(all_results, dataset_name="dataset"):
    """
    Generate all comparison plots for all metrics.

    Args:
        all_results: dict mapping method_name -> results
        dataset_name: name of dataset
    """
    ensure_dirs()

    metric_configs = [
        ('entropy', 'Entropy (higher = more IID)'),
        ('kl', 'KL Divergence (lower = more IID)'),
        ('dominant_classes', 'Dominant Classes for 80%'),
    ]

    for metric_name, ylabel in metric_configs:
        save_path = str(
            FIGURES_DIR /
            f"{dataset_name}_comparison_{metric_name}.png"
        )
        plot_methods_comparison(
            all_results, metric_name, ylabel,
            dataset_name=dataset_name, save_path=save_path
        )


def create_comparison_report(labels, num_classes, dataset_name, num_clients=10, seed=42):
    """
    Run full analysis and generate all comparison figures.

    Args:
        labels: numpy array of labels
        num_classes: number of classes
        dataset_name: name of dataset
        num_clients: number of clients
        seed: random seed

    Returns:
        dict with all results
    """
    print(f"\n{'=' * 60}")
    print(f"Comparison Report: {dataset_name.upper()}")
    print(f"{'=' * 60}")

    # Run all methods
    all_results = run_all_methods(labels, num_classes, num_clients, seed)

    # Generate per-method plots
    print("\nGenerating per-method plots...")
    for method_name, results in all_results.items():
        plot_all_metrics_for_method(results, dataset_name)

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_all_comparisons(all_results, dataset_name)

    # Save results summary
    summary = {
        'dataset': dataset_name,
        'num_clients': num_clients,
        'num_classes': num_classes,
        'num_samples': len(labels),
        'methods': {}
    }

    for method_name, results in all_results.items():
        summary['methods'][method_name] = {
            'param_name': results['param_name'],
            'param_values': [float(p) for p in results['param_values']],
            'metrics': {
                str(k): v for k, v in results['metrics'].items()
            }
        }

    summary_path = str(RESULTS_DIR / f"{dataset_name}_comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    return all_results


def main():
    """Main entry point: run comparison on MNIST and CIFAR-10."""
    ensure_dirs()

    # MNIST
    print("\nLoading MNIST dataset...")
    train_dataset, _ = dataset.load_mnist()
    mnist_labels = np.array(train_dataset.targets)
    mnist_results = create_comparison_report(mnist_labels, 10, "mnist")

    # CIFAR-10
    print("\nLoading CIFAR-10 dataset...")
    train_dataset, _ = dataset.load_cifar10()
    cifar_labels = np.array(train_dataset.targets)
    cifar_results = create_comparison_report(cifar_labels, 10, "cifar10")

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)

    return mnist_results, cifar_results


if __name__ == "__main__":
    main()
