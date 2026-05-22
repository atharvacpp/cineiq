"""
CineIQ MLflow Tracking Utility
================================
Handles MLflow experiment logging, model artifact tracking,
and leaderboard comparison for SVD training runs.
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the tracking URI dynamically so it saves to the root mlruns folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

try:
    import mlflow
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Tracking functionality disabled.")


def log_svd_experiment(experiment_name: str, params: dict, metrics: dict, model_path: str):
    """
    Logs hyperparameters, metrics, and model artifacts to MLflow.

    Args:
        experiment_name: Name of the MLflow experiment.
        params: Dict of hyperparameters (n_factors, n_epochs, lr_all).
        metrics: Dict of evaluation metrics (RMSE, precision@k, recall@k).
        model_path: Path to the serialized .pkl model file.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Skipping experiment logging.")
        return False

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        logger.info(f"Logging SVD experiment '{experiment_name}' to MLflow...")

        # Log hyperparameters (n_factors, n_epochs, lr_all)
        mlflow.log_params(params)

        # Log evaluation metrics (RMSE, precision@k, recall@k)
        mlflow.log_metrics(metrics)

        # Log model artifacts (serialized .pkl files)
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path)
            logger.info(f"Artifact {os.path.basename(model_path)} logged successfully.")
        else:
            logger.warning(f"Model file not found at {model_path}, artifact not logged.")

    return True


def compare_runs(experiment_name: str = "CineIQ_Collaborative_Filtering"):
    """
    Fetches all runs for the given experiment and prints a leaderboard
    sorted by RMSE (lower is better).

    Returns:
        pandas.DataFrame or None: Leaderboard DataFrame.
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Cannot compare runs.")
        return None

    try:
        import pandas as pd

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experiment '{experiment_name}' not found.")
            return None

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if runs.empty:
            print("No runs found for this experiment.")
            return None

        # Select important columns for the leaderboard
        cols = ['run_id']
        for c in ['params.n_factors', 'params.n_epochs', 'params.lr_all',
                   'metrics.rmse', 'metrics.precision_at_5', 'metrics.recall_at_5']:
            if c in runs.columns:
                cols.append(c)

        leaderboard = runs[cols].copy()

        # Sort by RMSE (lower is better)
        if 'metrics.rmse' in leaderboard.columns:
            leaderboard = leaderboard.sort_values(by='metrics.rmse', ascending=True).reset_index(drop=True)

        print("\n" + "=" * 70)
        print(f"🏆 MLFLOW LEADERBOARD: {experiment_name} 🏆")
        print("=" * 70)
        print(leaderboard.to_string(index=False))
        print("=" * 70 + "\n")

        return leaderboard
    except Exception as e:
        logger.error(f"Failed to fetch MLflow runs: {e}")
        return None


if __name__ == "__main__":
    compare_runs()
