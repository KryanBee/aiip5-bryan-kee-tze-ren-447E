import pandas as pd
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE

def split_data():
    """Splits processed data into training and testing sets and saves them."""
    print("[INFO] Loading processed data for train-test split...")
    df = pd.read_pickle("data/processed_data.pkl")

    # Splitting data into train and test
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Save train and test sets
    train_df.to_pickle("data/train.pkl")
    test_df.to_pickle("data/test.pkl")

    print("[INFO] Data successfully split and saved.")
    print(f"[INFO] Training data size: {train_df.shape}, Testing data size: {test_df.shape}")

if __name__ == "__main__":
    split_data()
