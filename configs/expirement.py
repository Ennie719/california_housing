import yaml
from sklearn.datasets import fetch_california_housing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from src.model_training import model_training
from src.preprocessing import splitting_data, load_data
import mlflow
import mlflow.sklearn
from src.model_evaluation import model_evaluation
from src.preprocessing import load_config, load_data, splitting_data
def trial_experiment(): 
    
    with open("configs/expirement.yaml") as f:
        config = yaml.safe_load(f)
    df = load_data()
    x_train, x_test, y_train, y_test =  splitting_data(df)
    
    mlflow.set_experiment("California Housing")
    for exp in config["model"]:
        # params = exp["model"]
        print("Starting expirements")
        print(f"{'='*60}")
        with mlflow.start_run():
            # check_train = model_training(df, "configs/config.yaml")
            model = model_training(df, "configs/expirement.yaml")
            # mse, r2 = model_evaluation(model, x_test, y_test)
            predictions = model.predict(x_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({"mse": mse, "r2": r2})
            mlflow.sklearn.log_model(model, "model")
            print(model["name"])
            print(model, f"Model")
            print(mse, f"MSE")
            print(r2, f"R square")
            print(predictions, f"Prediction")
    return 

if __name__ == "__main__":
    trial_experiment()
