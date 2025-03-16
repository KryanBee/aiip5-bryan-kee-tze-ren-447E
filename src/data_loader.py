import pandas as pd
import sqlite3
from config import DATA_PATH, TABLE_NAME, EXCLUDE_VARS

def load_data():
    """Load dataset from SQLite database into a Pandas DataFrame, removing excluded columns."""
    print("[INFO] Loading data from SQLite database...")
    conn = sqlite3.connect(DATA_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME};", conn)
    conn.close()

    # Drop excluded columns
    df.drop(columns=EXCLUDE_VARS, errors="ignore", inplace=True)

    print("[INFO] Data successfully loaded. Preview:")
    print(df.info())

    return df

if __name__ == "__main__":
    df = load_data()
    df.to_pickle("data/subsetted_data.pkl")
    print("[INFO] Subsetted data saved.")
