import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.preprocessing import load_data, splitting_data
from src.model_evaluation import model_evaluation
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(f"Duplicate values of dataset: {df.duplicated().sum()}")
print(f"Missing value of dataset: {df.isnull().sum()}")
def model_training(df):
    """
    Train a Random Forest Regressor on the training data.
    Returns:
    The trained model.
    """
    x_train,x_test, y_train,y_test = splitting_data(df)
    rf = RandomForestRegressor()
    rf_fit = rf.fit(x_train, y_train)
    return rf_fit

def feature_importance(rf, feature_names):
    """
    Calculate feature importance from the fitted Random Forest model.
    Returns:
    DataFrame containing features and their importance scores.
    """
    if rf is None or feature_names is None:
        raise ValueError("Model or feature names are None. Please provide valid inputs.")

    importances_val = rf.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances_val,
        'importance_Percent': importances_val * 100
    }).sort_values('importance', ascending=False)

    return importance_df