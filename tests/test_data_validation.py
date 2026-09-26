# AT LEAST 3 FUNCTION
import pandas as pd
import sys
import yaml
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from src.preprocessing import load_data, splitting_data
from src.model_training import model_training, feature_importance
from src.model_evaluation import model_evaluation
from sklearn.datasets import fetch_california_housing

@pytest.fixture
def df():
    return load_data()

def test_count_columns(df):
    assert len(df.columns) == 9, f"Expected 9 columns"

def test_features_check(df):
    non_numeric_cols = df.select_dtypes(exclude="number").columns
    assert len(non_numeric_cols) == 0, (
                f"Expected all columns to be numeric; found non-numeric columns: "
                f"{list(non_numeric_cols)}"
            )

def test_target_value(df):
    assert (df["MedHouseVal"] >= 0).all(), "The target values must be non-negative."

def test_feature_imp(df):
    """Test feature importance function works."""
    model = model_training(df, "configs/config.yaml")

    feature_names = df.drop(columns=["MedHouseVal"]).columns
    feat_imp = feature_importance(model, feature_names)
    
    # assert rf.empty, "<Data frame is empty.>"
    assert len(feature_names) > 0, "The feature names empty."
    assert not feat_imp.empty, "< The dataset is invalid.>"
    return feat_imp
# print(test_feature_imp())

def test_model(df):
    """
    Test the model_evaluation function with a simple linear regression model.
    """
    x_train,x_test, y_train,y_test = splitting_data(df)
    model = model_training(df, "configs/config.yaml")
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    assert mse >= 0
    assert rmse >= 0
    return mse, rmse
# df_test = test_model()
# print(df_test)