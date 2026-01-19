"""
Load and analyze federated learning partition metrics from JSON result files.
"""

import json
import pandas as pd
from pathlib import Path

# Project root is the parent of the src directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def load_results(results_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all metrics JSON files from the results directory.

    Args:
        results_dir: Path to results directory. Defaults to PROJECT_ROOT/results.

    Returns:
        tuple: (aggregate_df, per_client_df)
            - aggregate_df: Summary metrics for each partitioning configuration
            - per_client_df: Per-client metrics for detailed analysis
    """
    results_path = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    json_files = list(results_path.glob("*_metrics.json"))

    aggregate_records = []
    per_client_records = []

    for json_file in sorted(json_files):
        with open(json_file) as f:
            data = json.load(f)

        params = data["parameters"]
        agg = data["aggregate"]

        # Build aggregate record
        agg_record = {
            "file": json_file.name,
            "method": params["method"],
            "num_clients": params["num_clients"],
            "seed": params.get("seed"),
            # Method-specific parameters
            "alpha": params.get("alpha"),
            "shards_per_client": params.get("shards_per_client"),
            # Entropy metrics
            "entropy_mean": agg["entropy"]["mean"],
            "entropy_std": agg["entropy"]["std"],
            "entropy_min": agg["entropy"]["min"],
            "entropy_max": agg["entropy"]["max"],
            "entropy_max_possible": agg["entropy"]["max_possible"],
            # KL divergence metrics
            "kl_div_mean": agg["kl_divergence"]["mean"],
            "kl_div_std": agg["kl_divergence"]["std"],
            "kl_div_min": agg["kl_divergence"]["min"],
            "kl_div_max": agg["kl_divergence"]["max"],
            # Dominant classes
            "dominant_classes_mean": agg["dominant_classes"]["mean"],
            "dominant_classes_std": agg["dominant_classes"]["std"],
            # Sample counts
            "samples_mean": agg["sample_counts"]["mean"],
            "samples_std": agg["sample_counts"]["std"],
            "samples_min": agg["sample_counts"]["min"],
            "samples_max": agg["sample_counts"]["max"],
            "samples_total": agg["sample_counts"]["total"],
            "num_classes": agg["num_classes"],
        }
        aggregate_records.append(agg_record)

        # Build per-client records
        for client_id, client_data in data["per_client"].items():
            client_record = {
                "file": json_file.name,
                "method": params["method"],
                "alpha": params.get("alpha"),
                "shards_per_client": params.get("shards_per_client"),
                "client_id": int(client_id),
                "entropy": client_data["entropy"],
                "kl_divergence": client_data["kl_divergence"],
                "dominant_classes": client_data["dominant_classes"],
                "num_samples": client_data["num_samples"],
            }
            per_client_records.append(client_record)

    aggregate_df = pd.DataFrame(aggregate_records)
    per_client_df = pd.DataFrame(per_client_records)

    return aggregate_df, per_client_df


def get_comparison_table(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simplified comparison table sorted by heterogeneity.
    Lower entropy = more non-IID (heterogeneous).
    """
    df = aggregate_df.copy()

    # Create a readable config column
    def config_label(row):
        if row["method"] == "dirichlet":
            return f"Dirichlet (α={row['alpha']})"
        else:
            return f"Sharding ({row['shards_per_client']} shards)"

    df["config"] = df.apply(config_label, axis=1)

    # Select key columns for comparison
    comparison = df[[
        "config", "method", "entropy_mean", "entropy_std",
        "kl_div_mean", "dominant_classes_mean", "samples_std"
    ]].copy()

    # Sort by entropy (lower = more non-IID)
    comparison = comparison.sort_values("entropy_mean")

    return comparison


if __name__ == "__main__":
    # Load all results
    aggregate_df, per_client_df = load_results()

    print("=" * 70)
    print("AGGREGATE METRICS (all configurations)")
    print("=" * 70)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(aggregate_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("COMPARISON TABLE (sorted by heterogeneity - lower entropy = more non-IID)")
    print("=" * 70)
    comparison = get_comparison_table(aggregate_df)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 70)
    print("PER-CLIENT METRICS (first 20 rows)")
    print("=" * 70)
    print(per_client_df.head(20).to_string(index=False))

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS BY METHOD")
    print("=" * 70)
    print(aggregate_df.groupby("method")[["entropy_mean", "kl_div_mean", "samples_std"]].describe())
