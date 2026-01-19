"""
Non-IID Metrics Module for Federated Learning Data Partitioning

Provides functions to measure the degree of non-IID-ness in client data distributions.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy


def class_distribution(labels, num_classes):
    """
    Compute the class distribution (probability vector) for a set of labels.

    Args:
        labels: array of class labels
        num_classes: total number of classes

    Returns:
        numpy array of shape (num_classes,) with probabilities
    """
    labels = np.array(labels)
    counts = np.bincount(labels, minlength=num_classes)
    return counts / counts.sum() if counts.sum() > 0 else counts.astype(float)


def class_entropy(labels, num_classes):
    """
    Compute Shannon entropy of the class distribution.

    Higher entropy = more uniform distribution (more IID-like).
    Max entropy for K classes = log(K).

    Args:
        labels: array of class labels
        num_classes: total number of classes

    Returns:
        float: Shannon entropy in nats
    """
    dist = class_distribution(labels, num_classes)
    return scipy_entropy(dist)


def kl_divergence(client_dist, global_dist, epsilon=1e-10):
    """
    Compute KL divergence from global distribution to client distribution.

    KL(P||Q) = sum(P * log(P/Q))

    Higher KL divergence = client distribution more different from global.

    Args:
        client_dist: client's class probability distribution
        global_dist: global class probability distribution
        epsilon: small value to avoid log(0)

    Returns:
        float: KL divergence
    """
    client_dist = np.array(client_dist) + epsilon
    global_dist = np.array(global_dist) + epsilon

    # Normalize to ensure valid probability distributions
    client_dist = client_dist / client_dist.sum()
    global_dist = global_dist / global_dist.sum()

    return np.sum(client_dist * np.log(client_dist / global_dist))


def num_dominant_classes(labels, num_classes, threshold=0.8):
    """
    Count the minimum number of classes needed to reach a threshold of total samples.

    Lower count = more concentrated/non-IID distribution.

    Args:
        labels: array of class labels
        num_classes: total number of classes
        threshold: fraction of samples to reach (default 0.8 = 80%)

    Returns:
        int: number of classes needed to reach threshold
    """
    dist = class_distribution(labels, num_classes)
    sorted_dist = np.sort(dist)[::-1]  # Sort descending

    cumsum = 0
    for i, p in enumerate(sorted_dist):
        cumsum += p
        if cumsum >= threshold:
            return i + 1

    return num_classes


def compute_client_metrics(client_indices, all_labels, num_classes, global_dist=None):
    """
    Compute all metrics for a single client.

    Args:
        client_indices: array of data indices for this client
        all_labels: array of all labels in the dataset
        num_classes: total number of classes
        global_dist: global class distribution (computed if None)

    Returns:
        dict with metrics: entropy, kl_divergence, dominant_classes, num_samples
    """
    client_labels = all_labels[client_indices]
    client_dist = class_distribution(client_labels, num_classes)

    if global_dist is None:
        global_dist = class_distribution(all_labels, num_classes)

    return {
        'entropy': float(class_entropy(client_labels, num_classes)),
        'kl_divergence': float(kl_divergence(client_dist, global_dist)),
        'dominant_classes': int(num_dominant_classes(client_labels, num_classes)),
        'num_samples': len(client_indices),
        'class_distribution': client_dist.tolist()
    }


def compute_all_metrics(client_data_map, labels, num_classes):
    """
    Compute metrics for all clients and aggregate statistics.

    Args:
        client_data_map: dict mapping client_id -> array of indices
        labels: array of all labels in the dataset
        num_classes: total number of classes

    Returns:
        dict with per_client metrics and aggregate statistics
    """
    labels = np.array(labels)
    global_dist = class_distribution(labels, num_classes)

    # Compute per-client metrics
    per_client = {}
    entropies = []
    kl_divs = []
    dom_classes = []
    sample_counts = []

    for client_id, indices in client_data_map.items():
        metrics = compute_client_metrics(indices, labels, num_classes, global_dist)
        per_client[client_id] = metrics

        entropies.append(metrics['entropy'])
        kl_divs.append(metrics['kl_divergence'])
        dom_classes.append(metrics['dominant_classes'])
        sample_counts.append(metrics['num_samples'])

    # Compute aggregate statistics
    max_entropy = np.log(num_classes)  # Maximum possible entropy

    aggregate = {
        'entropy': {
            'mean': float(np.mean(entropies)),
            'std': float(np.std(entropies)),
            'min': float(np.min(entropies)),
            'max': float(np.max(entropies)),
            'max_possible': float(max_entropy)
        },
        'kl_divergence': {
            'mean': float(np.mean(kl_divs)),
            'std': float(np.std(kl_divs)),
            'min': float(np.min(kl_divs)),
            'max': float(np.max(kl_divs))
        },
        'dominant_classes': {
            'mean': float(np.mean(dom_classes)),
            'std': float(np.std(dom_classes)),
            'min': int(np.min(dom_classes)),
            'max': int(np.max(dom_classes))
        },
        'sample_counts': {
            'mean': float(np.mean(sample_counts)),
            'std': float(np.std(sample_counts)),
            'min': int(np.min(sample_counts)),
            'max': int(np.max(sample_counts)),
            'total': int(np.sum(sample_counts))
        },
        'num_clients': len(client_data_map),
        'num_classes': num_classes,
        'global_distribution': global_dist.tolist()
    }

    return {
        'per_client': per_client,
        'aggregate': aggregate
    }


def print_metrics_summary(metrics):
    """
    Print a formatted summary of computed metrics.

    Args:
        metrics: dict returned by compute_all_metrics
    """
    agg = metrics['aggregate']

    print("=" * 60)
    print("Partition Metrics Summary")
    print("=" * 60)
    print(f"Number of clients: {agg['num_clients']}")
    print(f"Number of classes: {agg['num_classes']}")
    print(f"Total samples: {agg['sample_counts']['total']}")
    print()

    print("Entropy (higher = more IID):")
    print(f"  Mean: {agg['entropy']['mean']:.4f} (max possible: {agg['entropy']['max_possible']:.4f})")
    print(f"  Std:  {agg['entropy']['std']:.4f}")
    print(f"  Range: [{agg['entropy']['min']:.4f}, {agg['entropy']['max']:.4f}]")
    print()

    print("KL Divergence (lower = more IID):")
    print(f"  Mean: {agg['kl_divergence']['mean']:.4f}")
    print(f"  Std:  {agg['kl_divergence']['std']:.4f}")
    print(f"  Range: [{agg['kl_divergence']['min']:.4f}, {agg['kl_divergence']['max']:.4f}]")
    print()

    print("Dominant Classes for 80% (lower = more non-IID):")
    print(f"  Mean: {agg['dominant_classes']['mean']:.2f}")
    print(f"  Range: [{agg['dominant_classes']['min']}, {agg['dominant_classes']['max']}]")
    print()

    print("Samples per Client:")
    print(f"  Mean: {agg['sample_counts']['mean']:.1f}")
    print(f"  Std:  {agg['sample_counts']['std']:.1f}")
    print(f"  Range: [{agg['sample_counts']['min']}, {agg['sample_counts']['max']}]")
    print("=" * 60)
