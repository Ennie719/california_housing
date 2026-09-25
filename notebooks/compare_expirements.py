from ast import main

import mlflow

def get_all_experiment_ids():
    """Return IDs for all active MLflow experiments."""
    experiments = mlflow.search_experiments()
    return [
        experiment.experiment_id
        for experiment in experiments
        if experiment.lifecycle_stage == "active"
    ]

def search_runs():
    """Return completed runs across all active experiments."""
    all_experiment_ids = get_all_experiment_ids()
    return mlflow.search_runs(
        experiment_ids=all_experiment_ids,
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
    )

def best_run():
    """Find out which model performs the best."""
    experiment = mlflow.get_experiment_by_name("California Housing")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.r2_score DESC"],
        # best_run =[]
    )
    return runs.iloc[0]
best_run()

# print(search_runs())
