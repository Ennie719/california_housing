import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.preprocessing import load_data, splitting_data
from src.model_training import model_training, feature_importance
from sklearn.datasets import fetch_california_housing
import pandas as pd
from src.model_evaluation import model_evaluation
housing = fetch_california_housing(as_frame=True)
df = housing.frame
np.random.seed(42)
n_rows = 200
model = model_training(df)

df_sample = pd.DataFrame({
    "size_sqft": np.random.randint(500, 3500, n_rows),
    "num_rooms": np.random.randint(1, 6, n_rows),
    "age_years": np.random.randint(0, 50, n_rows),
})

# Target: house price (float), based on features + noise
df_sample["MedHouseVal"] = (
    df_sample["size_sqft"] * 150
    + df_sample["num_rooms"] * 5000
    - df_sample["age_years"] * 300
    + np.random.normal(0, 10000, n_rows)
).round(2)  # keep target as float

# # ---- 2. Split features (X) and target (y) ----
# X = df_sample[["size_sqft", "num_rooms", "age_years"]]
# y = df_sample["price"]
sample_splitt_data = (df_sample)
splitting_data_sample = splitting_data(df_sample)
# splitting_data_sample
sample_model = model_training(df_sample)
# def model_evaluation(model, x_test, y_test):
x_train, x_test, y_train, y_test = splitting_data(df_sample)
sample_model_eval = model_evaluation(sample_model, x_test, y_test)
sample_model_eval