from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Returns:
    The California housing data as a DataFrame.
    """  
    return fetch_california_housing(as_frame=True).frame
# df = load_data()
# print(df.info())
def encode_categoricals(df, columns):
    """One-hot encode categorical columns."""
    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)
    return df
# df = encode_categoricals(load_data(),['HouseAge'])
# print(df.head())
def check_missing_values(df):
    """
    Check for missing values in the DataFrame.
    Returns:
    A boolean indicating if there are missing values.
    """
    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")
    
    return df.isnull().values.any()
def check_duplicates(df):
    """
    Check for duplicate rows in the DataFrame.
    Returns:
    A boolean indicating if there are duplicate rows.
    """
    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")
    
    return df.duplicated().any()

def splitting_data(df):
    """
    Split the input DataFrame into training and validation data.
    Returns:
    Training and validation features and targets.
    """
    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")
    
    target = df['MedHouseVal']
    features = df.drop('MedHouseVal', axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )

    return x_train, x_test, y_train, y_test

# df = splitting_data(load_data())
# print(f"Split data shapes - X_train: {df[0].shape[0]}, X_test: {df[1].shape[0]}, y_train: {df[2].shape[0]}, y_test: {df[3].shape[0]}")
def scale_features(x_train, x_test):
    """
    Scale the features using StandardScaler.
    Returns:
    Scaled training and validation features.
    """
    if x_train is None or x_test is None:
        raise ValueError("Training or test features are None. Please provide valid inputs.")
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled
# df = load_data()
# df_scaled = encode_categoricals(df, ['HouseAge'])
# print(f"Feature scaling completed. Scaled DataFrame shape: {df_scaled.shape}")
def feature_target_corr():
    """
    Calculate the correlation between features and the target variable.
    Returns:
    A DataFrame containing features and their correlation with the target.
    """
    df = load_data()
    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")
    
    corr_matrix = df.corr()
    target_corr = corr_matrix['MedHouseVal'].drop('MedHouseVal')
    importance_df = target_corr[target_corr > 0.1].sort_values(ascending=False)
    return importance_df
# df = feature_target_corr()
# print(f"Feature-target correlation:\n{df}")