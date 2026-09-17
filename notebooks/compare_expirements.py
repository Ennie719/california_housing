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


print(search_runs())