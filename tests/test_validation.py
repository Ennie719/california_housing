# AT LEAST 3 FUNCTION
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from src.preprocessing import load_data, splitting_data
from src.model_training import model_training, feature_importance
from src.model_evaluation import model_evaluation
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame
model = model_training(df)

# df.columns.list
# columns = len(df.columns)
# print(columns)
def test_count_columns():
    assert len(df.columns) == 9, f"Expected 9 columns"
    
def test_features_check():
    assert df.columns > 0, f"columns value has to positive"

def test_target_value():
    assert df['MedHouseVal'] >= 0, f"The target has to positive"


def test_feature_imp():
    """Test feature importance function works."""
    feature_names = df.drop(columns=["MedHouseVal"]).columns
    feat_imp = feature_importance(model, feature_names)
    
    # assert rf.empty, "<Data frame is empty.>"
    assert len(feature_names) > 0, "The feature names empty."
    assert not feat_imp.empty, "< The dataset is invalid.>"
    return feat_imp
# df = load_data()
# df_test_feat_imp = model_training(df)
# test = test_feature_imp(df_test_feat_imp, x_train.columns)
# print(test) 
def test_model():
    """
    Test the model_evaluation function with a simple linear regression model.
    """
    x_train,x_test, y_train,y_test = splitting_data(df)
    model = model_training(df)
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    assert mse >= 0
    assert rmse >= 0
    return mse, rmse
df_test = test_model()
print(df_test)