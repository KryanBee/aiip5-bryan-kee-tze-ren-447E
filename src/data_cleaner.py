import pandas as pd
from config import QUANTITATIVE_VARS, CATEGORICAL_VARS, NEGATIVE_TO_NA, IMPUTE_METHOD

def clean_data(df):
    """Cleans the dataset: processes categorical and quantitative columns."""
    print("[INFO] Cleaning dataset...")
    
    # Standardize categorical variables (convert to lowercase)
    df[CATEGORICAL_VARS] = df[CATEGORICAL_VARS].apply(lambda x: x.str.lower())

    # Clean quantitative variables
    df[QUANTITATIVE_VARS] = df[QUANTITATIVE_VARS].replace(["None", "nan", "", "N/A"], pd.NA)
    df[QUANTITATIVE_VARS] = df[QUANTITATIVE_VARS].astype(str).replace(r"\s*ppm", "", regex=True)
    df[QUANTITATIVE_VARS] = df[QUANTITATIVE_VARS].apply(pd.to_numeric, errors="coerce")
    
    # Replace negative values with NA if NEGATIVE_TO_NA is True
    if NEGATIVE_TO_NA:
        df[QUANTITATIVE_VARS] = df[QUANTITATIVE_VARS].map(lambda x: pd.NA if x < 0 else x)
    
    # Handle missing values based on IMPUTE_METHOD
    if IMPUTE_METHOD == "mean":
        df[QUANTITATIVE_VARS] = df[QUANTITATIVE_VARS].fillna(df[QUANTITATIVE_VARS].mean())
    elif IMPUTE_METHOD == "median":
        df[QUANTITATIVE_VARS] = df[QUANTITATIVE_VARS].fillna(df[QUANTITATIVE_VARS].median())
    elif IMPUTE_METHOD == "mode":
        df[QUANTITATIVE_VARS] = df[QUANTITATIVE_VARS].fillna(df[QUANTITATIVE_VARS].mode().iloc[0])
    elif IMPUTE_METHOD == "drop":
        df = df.dropna()

    print("[INFO] Data cleaning completed.")
    print(df.info())

    return df

if __name__ == "__main__":
    df = pd.read_pickle("data/subsetted_data.pkl")  # Load from previous step
    df = clean_data(df)
    df.to_pickle("data/cleaned_data.pkl")  # Saves cleaned DataFrame
    print("[INFO] Cleaned data saved.")