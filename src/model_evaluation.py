import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from src.model_training import model_training
from src.preprocessing import splitting_data, load_data

def model_evaluation(model, x_test, y_test):
    """
    Evaluate the performance of a Random Forest Regressor on the test data.
    Returns:
    The trained model and its performance metrics.
    """
    predictions = model.predict(x_test)
    mse_rf_tuned = mean_squared_error(y_test, predictions)
    rmse_rf_tuned = np.sqrt(mse_rf_tuned)
    r2_rf_tuned = r2_score(y_test, predictions)
    # print("Metrics:")
    # print(f"MSE:  {mse_rf_tuned:.4f}")
    # # print(f"RMSE: {rmse_rf_tuned:.4f}")
    # print(f"R²:   {r2_rf_tuned:.1%}")
    # print(type(predictions))
    # print()
    return mse_rf_tuned, r2_rf_tuned
# if name == "main":
df = load_data()
# check_train = model_training(df, "configs/config.yaml")

df_rf= model_training(df, "configs/config.yaml")
x_train, x_test, y_train, y_test =  splitting_data(df)
# test_size=0.25, random_state=42)
test = model_evaluation(df_rf, x_test, y_test)
test