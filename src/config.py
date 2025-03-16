import numpy as np
import pandas as pd

### DATA SETTINGS ###
DATA_PATH = "data/agri.db" # Ensure the agri.db is placed in the correct folder
TABLE_NAME = "farm_data"
DATA_FORMAT = "sqlite"

### DATA CLEANING ###
NEGATIVE_TO_NA = True  # Convert negative values to NA
IMPUTE_METHOD = "median"  # Options: "mean", "median", "mode", "drop"
EXCLUDE_VARS = ["System Location Code", "Previous Cycle Plant Type"] # Variables to exclude from modeling
CATEGORICAL_VARS = ["Plant Type", "Plant Stage"] # Categorical variables to include for modeling
QUANTITATIVE_VARS = [ # Quantitative variables to include for modeling
    "Temperature Sensor (°C)", 
    "Light Intensity Sensor (lux)", 
    "Humidity Sensor (%)", 
    "Water Level Sensor (mm)",
    "CO2 Sensor (ppm)", 
    "EC Sensor (dS/m)", 
    "O2 Sensor (ppm)", 
    "pH Sensor",
    "Nutrient N Sensor (ppm)", 
    "Nutrient P Sensor (ppm)", 
    "Nutrient K Sensor (ppm)"
]

### FEATURE ENGINEERING ###
BINNING_RULES = {
    "Light Intensity Sensor (lux)": {
        "bins": [-float("inf"), 200, 400, float("inf")],
        "labels": ["Low", "Medium", "High"]
    },
    "CO2 Sensor (ppm)": {
        "bins": [-float("inf"), 1000, 1200, float("inf")],
        "labels": ["Low", "Medium", "High"]
    },
    "Humidity Sensor (%)": lambda x: "Low" if pd.notna(x) and x < 70 else ("High" if pd.notna(x) else np.nan),
    "Nutrient N Sensor (ppm)": lambda x: "Low" if pd.notna(x) and x < 150 else ("High" if pd.notna(x) else np.nan),
    "Nutrient P Sensor (ppm)": lambda x: "Low" if pd.notna(x) and x < 50 else ("High" if pd.notna(x) else np.nan),
    "Nutrient K Sensor (ppm)": lambda x: "Low" if pd.notna(x) and x < 200 else ("High" if pd.notna(x) else np.nan),
}

### TRAIN-TEST SPLIT ###
TEST_SIZE = 0.2  # Percentage for test set
RANDOM_STATE = 42  # For reproducibility

### MODEL TRAINING ###
HYPERPARAM_GRID = {
    "random_forest": {
        "n_estimators": [100, 150],
        "max_depth": [10, 15],
        "min_samples_split": [10, 20], 
        "min_samples_leaf": [2, 4]
    },
    "xgboost": {
        "n_estimators": [100, 150],
        "learning_rate": [0.01, 0.05],
        "max_depth": [5, 8], 
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8]
    }
}
CV_FOLDS = 5  # Number of folds for cross-validation
HYPERPARAMETER_TUNING = True  # Set to False to disable tuning