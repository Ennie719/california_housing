from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml


def load_config(path):
    """Load config yaml file"""
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
    
def load_data():
    """
    Returns:
    The California housing data as a DataFrame.
    """  
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
        df = fetch_california_housing(as_frame=config["data"]["as_frame"]).frame
    return df
    # df = fetch_california_housing(as_frame=True).frame
    # return df.drop_duplicates().reset_index(drop=True)
# if __name__ == "__main__":
#     df = load_data()
def encode_categoricals(df, columns):
    """One-hot encode categorical columns."""
    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)
    return df

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
with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
def splitting_data(df, test_size=config["test_size"], random_state=config["random_state"]):
    """
    Split the input DataFrame into training and validation data.
    Returns:
    Training and validation features and targets.
    """
    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")
    # print(df)
    target = df['MedHouseVal']
    features = df.drop('MedHouseVal', axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test
if __name__ == "__main__":

    df = load_data()
    df_split = splitting_data(df)
    print(df_split)
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
def feature_target_corr(df):
    """
    Calculate the correlation between features and the target variable.
    Returns:
    A DataFrame containing features and their correlation with the target.
    """
    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid DataFrame.")
    
    corr_matrix = df.corr()
    target_corr = corr_matrix['MedHouseVal'].drop('MedHouseVal')
    importance_df = target_corr[target_corr > 0.1].sort_values(ascending=False)
    return importance_df
