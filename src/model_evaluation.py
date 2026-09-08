# missing values
from pyexpat import model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import numpy as np
from preprocessing import load_data, dataframe_check, splitting_data, model_training, feature_importance
from model_training import model_training


def model_evaluation(x_valid , y_valid, x_train, y_train):
    """
    Evaluate the performance of a Random Forest Regressor on the validation data.
    Returns:
    The trained model and its performance metrics.
    """
    model = model_training(x_valid , y_valid, x_train, y_train)
    # Calculate performance metrics
    predictions_rf_tuned_val = model.predict(x_train)
    
    mse_rf_tuned = mean_squared_error(y_train, predictions_rf_tuned_val)
    rmse_rf_tuned = np.sqrt(mse_rf_tuned)
    r2_rf_tuned = model.score(x_train, y_train)
    print("Metrics:")
    print(f"MSE:  {mse_rf_tuned:.4f}")
    print(f"RMSE: {rmse_rf_tuned:.4f}")
    print(f"R²:   {r2_rf_tuned:.1%}")
    print()
    return mse_rf_tuned, rmse_rf_tuned, r2_rf_tuned