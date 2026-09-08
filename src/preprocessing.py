# missing values
from pyexpat import model
import sklearn
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.ensemble import RandomForestRegressor

def load_data(df):
    """
    df (pd.DataFrame): The DataFrame to validate.

    Returns:
    True if the DataFrame is valid, False otherwise.
    """  
    if df is None:
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")
    return df
def dataframe_check(df):
    """
    Validate the input DataFrame for duplicates and missing values.
    Returns:
    True if the DataFrame is clean, False otherwise.
    """
    if df is None:
        df = df.copy()  # Create a copy to avoid modifying the original DataFrame
        duplicate_count = df.duplicated().sum()
        missing_values_count = df.isnull().sum()
        raise ValueError(f"DataFrame is None. Duplicate values: {duplicate_count}, Missing values: {missing_values_count}")
    Returns: df
# feature scaling   
def splitting_data(df):
    """
    Perform feature scaling on the input DataFrame.
    Returns:
    Scaled features and target variable.
    """
    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")
    
    target = df['MedHouseVal']
    features = df.drop('MedHouseVal', axis=1)
    
    X_temp, X_test, y_temp, Y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )
    
    x_train, x_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    
    return x_train, x_valid, y_train, y_valid
# target = df['MedHouseVal']
# features = df.drop('MedHouseVal', axis=1)
# X_temp, X_test, y_temp, Y_test = train_test_split(
#     features, target, test_size=0.25, random_state=42
# )
# x_train, x_valid, y_train, y_valid = train_test_split(
#     X_temp, y_temp, test_size=0.25, random_state=42
# )
def model_traininggg(x_valid , y_valid, x_train, y_train):
    """
    Train a Random Forest Regressor on the training data.
    Returns:
    Trained model.
    """
    if x_train is None or y_train is None:
        raise ValueError("Training data is None. Please provide valid training data.")
    
    rf = RandomForestRegressor()
    rf.fit(x_valid, y_valid)
    
    return rf
def feature_importance(rf, feature_names):
    """
    Calculate feature importance from the trained Random Forest model.
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
# rf = RandomForestRegressor()
# rf.fit(x_train, y_train)
# # model.fit(x_train, y_train)  
# importance = rf.feature_importances_ 

# importance = pd.DataFrame({
#     'feature': feature_names,
#     'importance': importances_val,
#         'importance_Percent': importances_val * 100
# }).sort_values('importance', ascending=False)

# predictions = model.predict(x_valid)
# mse = mean_squared_error(y_valid, predictions)
# print(f"Mean Squared Error: {mse}")
# # print(f"Model Coefficients: {model.coef_}")
# std_scaler = StandardScaler()
# minmax_scaler = MinMaxScaler()
# robust_scaler = RobustScaler()
# df_standardized = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)
# df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df), columns=df.columns)
# df_robust = pd.DataFrame(robust_scaler.fit_transform(df), columns=df.columns)

# print("Standardized:\n", df_standardized)
# print("\nMin-Max Scaled:\n", df_minmax)
# print("\nRobust Scaled:\n", df_robust)

# result = permutation_importance(
#     model, x_valid, y_valid, n_repeats=10, random_state=42, n_jobs=-1

# )


# MinMaxScaler = df.copy()
# scaler = MinMaxScaler()
# MinMaxScaler = MinMaxScaler.fit_transform(MinMaxScaler[columns])
# print(f"MinMaxScaler: {MinMaxScaler}")

# categoical variables
# splitting the data into train and test sets