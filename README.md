# fl_research

Federated learning data partitioning tools.

## src/

| File | Description |
|------|-------------|
| `dataset.py` | Load datasets (MNIST, CIFAR-10) and partition them (Dirichlet, sharding) |
| `metrics.py` | Compute non-IID metrics (entropy, KL divergence, etc.) |
| `visualize_partitions.py` | Generate plots showing class distributions per client |
| `load_results.py` | Load and display metrics from JSON result files |
| `compare_methods.py` | Plot comparisons between partitioning methods |
| `partitioning_mnist.py` | Run partitioning experiments on MNIST |
| `partitioning_cifar.py` | Run partitioning experiments on CIFAR-10 |
| `test_partitioning.py` | Unit tests for partitioning functions |
