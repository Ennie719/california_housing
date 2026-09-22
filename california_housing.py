import mlflow
import mlflow.sklearn
from src.model_evaluation import model_evaluation
from src.model_training import model_training
from src.preprocessing import load_config, load_data, splitting_data
import json
import os
def main():
	config = load_config("configs/config.yaml")
	df = load_data()
	x_train, x_valid, y_train, y_valid = splitting_data(
		df,
  config["test_size"], config["random_state"]
	)
	model = model_training(df)

	mlflow.set_experiment("California Housing")
	with mlflow.start_run():
		mse, r2 = model_evaluation(model, x_valid, y_valid)
		mlflow.log_params(model.get_params())
		mlflow.log_metrics({"mse": mse, "r2": r2})
		mlflow.sklearn.log_model(model, "model")
    # return df
	stats ={"mse": mse, "r2": r2}

	os.makedirs("metrics", exist_ok=True)
	with open("metrics/results.json", "w") as f:
		json.dump(stats, f)
if __name__ == "__main__":
	main()