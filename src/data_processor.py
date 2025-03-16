import pandas as pd
from config import QUANTITATIVE_VARS, CATEGORICAL_VARS, BINNING_RULES

def process_data(df):
    """Performs feature engineering: binning and one-hot encoding."""
    print("[INFO] Processing dataset...")
    
    # Apply binning rules
    for col, rule in BINNING_RULES.items():
        if col in df.columns and col in QUANTITATIVE_VARS:  
            if isinstance(rule, dict):  # If rule is a dictionary with binning info
                df[f"{col}_bin"] = pd.cut(df[col], bins=rule["bins"], labels=rule["labels"], include_lowest=True)
            elif callable(rule):  # If rule is a function, apply it directly
                df[f"{col}_bin"] = df[col].apply(rule)
    
    # Drop original quantitative columns that were binned, if they exist in the dataframe
    df.drop(columns=[col for col in BINNING_RULES.keys() if col in df.columns], errors="ignore", inplace=True)
    
    # One-hot encode categorical and binned features
    categorical_features = CATEGORICAL_VARS + [f"{col}_bin" for col in BINNING_RULES.keys() if f"{col}_bin" in df.columns]
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    print("[INFO] Feature engineering completed.")
    print(df.info())

    return df

if __name__ == "__main__":
    df = pd.read_pickle("data/cleaned_data.pkl")  # Load from previous step
    df = process_data(df)
    df.to_pickle("data/processed_data.pkl")  # Saves processed DataFrame
    print("[INFO] Processed data saved.")