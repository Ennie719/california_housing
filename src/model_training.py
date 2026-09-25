import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.preprocessing import load_data, splitting_data
import yaml

def model_training(df, config_yaml):
    """
    Train a Random Forest Regressor on the training data.
    Returns:
    The trained model.
    """
    with open(config_yaml) as f:
        config = yaml.safe_load(f)
        params = config["model"]
        # print(params["n_estimators"])
        x_train,x_test, y_train,y_test = splitting_data(df)

        rf = RandomForestRegressor(n_estimators = params["n_estimators"],
                                max_depth = params["max_depth"],
                                min_samples_split = params["min_samples_split"])
        rf_fit = rf.fit(x_train, y_train)
    return rf_fit

if __name__ == "__main__":
    df = load_data()
    check_train = model_training(df, "configs/config.yaml")
    print(check_train)
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