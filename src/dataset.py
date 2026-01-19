import torch
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path

# Project root is the parent of the src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_mnist():
    """Load MNIST dataset with standard transforms."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def load_cifar10():
    """Load CIFAR-10 dataset with standard transforms."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


# =============================================================================
# Generalized Partitioning Functions (work with any dataset)
# =============================================================================


def partition_sharding(targets, num_clients=10, shards_per_client=2, seed=None):
    """
    Generalized sharding partitioning - works with any dataset.

    Sorts data by label, divides into shards, assigns random shards to clients.

    Args:
        targets: numpy array of labels (e.g., train_dataset.targets)
        num_clients: number of clients to partition data among
        shards_per_client: number of shards each client receives
        seed: random seed for reproducibility

    Returns:
        dict mapping client_id -> numpy array of indices
    """
    if seed is not None:
        np.random.seed(seed)

    targets = np.array(targets)
    num_shards = num_clients * shards_per_client
    shard_size = len(targets) // num_shards
    data_indices = np.arange(len(targets))

    # Sort indices by label
    sorted_indices = data_indices[np.argsort(targets)]

    # Create shards
    shards = [
        sorted_indices[i * shard_size : (i + 1) * shard_size]
        for i in range(num_shards)
    ]
    np.random.shuffle(shards)

    # Assign shards to clients
    client_data_map = {}
    for i in range(num_clients):
        assigned_shards = shards[i * shards_per_client : (i + 1) * shards_per_client]
        client_data_map[i] = np.concatenate(assigned_shards)

    return client_data_map


def partition_dirichlet(targets, num_clients=10, alpha=0.5, seed=None):
    """
    Generalized Dirichlet partitioning - works with any dataset.

    Samples class proportions for each client from Dir(alpha).
    Lower alpha = more non-IID.

    Args:
        targets: numpy array of labels
        num_clients: number of clients to partition data among
        alpha: Dirichlet concentration parameter
        seed: random seed for reproducibility

    Returns:
        dict mapping client_id -> numpy array of indices
    """
    if seed is not None:
        np.random.seed(seed)

    targets = np.array(targets)
    num_classes = len(np.unique(targets))
    client_data_map = {i: [] for i in range(num_clients)}
    idxs = np.arange(len(targets))

    for c in range(num_classes):
        class_idxs = idxs[targets == c]
        np.random.shuffle(class_idxs)

        # Sample Dirichlet proportions and split
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_idxs)).astype(int)[:-1]
        splits = np.split(class_idxs, proportions)

        for i in range(num_clients):
            client_data_map[i].extend(splits[i].tolist())

    # Convert lists to numpy arrays
    for i in range(num_clients):
        client_data_map[i] = np.array(client_data_map[i])

    return client_data_map


def partition_dirichlet_balanced(targets, num_clients=10, alpha=0.5, seed=None):
    """
    Balanced Dirichlet partitioning - enforces equal sample counts per client.

    Args:
        targets: numpy array of labels
        num_clients: number of clients to partition data among
        alpha: Dirichlet concentration parameter
        seed: random seed for reproducibility

    Returns:
        dict mapping client_id -> numpy array of indices
    """
    if seed is not None:
        np.random.seed(seed)

    targets = np.array(targets)
    num_classes = len(np.unique(targets))
    client_data_map = {i: [] for i in range(num_clients)}
    idxs = np.arange(len(targets))
    target_size = len(targets) // num_clients

    for c in range(num_classes):
        class_idxs = idxs[targets == c]
        np.random.shuffle(class_idxs)

        # Sample proportions with balancing constraint
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([
            p * (len(client_data_map[i]) < target_size)
            for i, p in enumerate(proportions)
        ])
        proportions /= proportions.sum()
        proportions = (np.cumsum(proportions) * len(class_idxs)).astype(int)[:-1]

        splits = np.split(class_idxs, proportions)
        for i in range(num_clients):
            client_data_map[i].extend(splits[i].tolist())

    # Convert lists to numpy arrays
    for i in range(num_clients):
        client_data_map[i] = np.array(client_data_map[i])

    return client_data_map


# =============================================================================
# Legacy MNIST-specific Functions (kept for backwards compatibility)
# =============================================================================


def partition_mnist_noniid_small(train_dataset, num_clients=10, shards_per_client=2):
    # num_shards = num_clients * shards_per_client
    num_shards = 100
    shard_size = len(train_dataset) // num_shards
    data_indices = np.arange(len(train_dataset))
    labels = np.array(train_dataset.targets)

    sorted_indices = data_indices[np.argsort(labels)]

    shards = [
        sorted_indices[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)
    ]
    np.random.shuffle(shards)

    client_data_map = {i: [] for i in range(num_clients)}
    for i in range(num_clients):
        assigned_shards = shards[i * shards_per_client : (i + 1) * shards_per_client]
        client_data_map[i] = np.concatenate(assigned_shards)

    return client_data_map


def partition_mnist_noniid_dict(train_dataset, num_clients=10, shards_per_client=2):
    num_shards = num_clients * shards_per_client  # Use ALL data
    shard_size = len(train_dataset) // num_shards  # Recompute shard size
    data_indices = np.arange(len(train_dataset))
    labels = np.array(train_dataset.targets)

    sorted_indices = data_indices[np.argsort(labels)]
    shards = [
        sorted_indices[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)
    ]
    np.random.shuffle(shards)

    client_data_map = {}
    for i in range(num_clients):
        assigned_shards = shards[i * shards_per_client : (i + 1) * shards_per_client]
        client_data_map[i] = np.concatenate(assigned_shards)
    return client_data_map


def partition_mnist_dirichlet_fair(train_dataset, num_clients=10, alpha=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    labels = np.array(train_dataset.targets)
    num_classes = len(np.unique(labels))
    client_data_map = {i: [] for i in range(num_clients)}
    idxs = np.arange(len(labels))

    # For each class, split indices among clients using Dirichlet distribution
    for c in range(num_classes):
        class_idxs = idxs[labels == c]
        np.random.shuffle(class_idxs)

        # Sample proportions for clients from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array(
            [
                p * (len(client_data_map[i]) < len(labels) / num_clients)
                for i, p in enumerate(proportions)
            ]
        )
        proportions /= proportions.sum()
        proportions = (np.cumsum(proportions) * len(class_idxs)).astype(int)[:-1]

        # Split class indices and assign to clients
        splits = np.split(class_idxs, proportions)
        for i in range(num_clients):
            client_data_map[i].extend(splits[i].tolist())

    return client_data_map


def partition_mnist_dirichlet(train_dataset, num_clients=10, alpha=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    labels = np.array(train_dataset.targets)
    num_classes = len(np.unique(labels))
    client_data_map = {i: [] for i in range(num_clients)}
    idxs = np.arange(len(labels))

    for c in range(num_classes):
        class_idxs = idxs[labels == c]
        np.random.shuffle(class_idxs)

        # Sample Dirichlet proportions and normalize
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_idxs)).astype(int)[:-1]
        splits = np.split(class_idxs, proportions)

        for i in range(num_clients):
            client_data_map[i].extend(splits[i].tolist())

    return client_data_map

