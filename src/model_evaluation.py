from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

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
    print("Metrics:")
    print(f"MSE:  {mse_rf_tuned:.4f}")
    print(f"RMSE: {rmse_rf_tuned:.4f}")
    print(f"R²:   {r2_rf_tuned:.1%}")
    print()
    return mse_rf_tuned, rmse_rf_tuned, r2_rf_tuned