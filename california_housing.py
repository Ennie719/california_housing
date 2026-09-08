from sklearn.datasets import fetch_california_housing

#  - MedInc        median income in block group
#     - HouseAge      median house age in block group
#     - AveRooms      average number of rooms per household
#     - AveBedrms     average number of bedrooms per household
#     - Population    block group population
#     - AveOccup      average number of household members
#     - Latitude      block group latitude
#     - Longitude     block group longitude
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def main():
	housing = fetch_california_housing(as_frame=True)
	features = housing.data
	target = housing.target
	x_train, x_test, y_train, y_test = train_test_split(
		features, target, test_size=0.25, random_state=42
	)

	model_params = {
		"n_estimators": 100,
		"max_depth": 10,
		"random_state": 42,
	}
	model = RandomForestRegressor(**model_params)

	with mlflow.start_run():
		model.fit(x_train, y_train)
		predictions = model.predict(x_test)

		mlflow.log_params(model_params)
		mlflow.log_metrics(
			{
				"mse": mean_squared_error(y_test, predictions),
				"rmse": mean_squared_error(y_test, predictions) ** 0.5,
				"r2": r2_score(y_test, predictions),
			}
		)
		mlflow.sklearn.log_model(model, "model")

		print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.4f}")
		print(f"R2: {r2_score(y_test, predictions):.4f}")


if __name__ == "__main__":
	main()