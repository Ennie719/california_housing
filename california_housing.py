import mlflow
import mlflow.sklearn
from src.model_evaluation import model_evaluation
from src.model_training import model_training
from src.preprocessing import dataframe_check, load_config, load_data, splitting_data

def main():
	config = load_config("configs/config.yaml")
	df = dataframe_check(load_data())
	x_train, x_valid, y_train, y_valid = splitting_data(
		df, config["test_size"], config["random_state"]
	)
	model = model_training(x_train, y_train, config["model"]["params"])

	mlflow.set_experiment("California Housing")
	with mlflow.start_run():
		mse, rmse, r2 = model_evaluation(model, x_valid, y_valid)
		mlflow.log_params(model.get_params())
		mlflow.log_metrics({"mse": mse, "rmse": rmse, "r2": r2})
		mlflow.sklearn.log_model(model, "model")
    # return df

if __name__ == "__main__":
	main()