import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from config import HYPERPARAMETER_TUNING, CV_FOLDS, HYPERPARAM_GRID

def tune_hyperparameters(model, param_grid, X_train, y_train):
    """Uses GridSearchCV to find the best hyperparameters."""
    print(f"[INFO] Performing GridSearchCV for {model.__class__.__name__}...")
    grid_search = GridSearchCV(model, param_grid, cv=CV_FOLDS, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"[INFO] Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_train, y_train):
    """Evaluate the model using the training data."""
    # Make predictions on the training set
    predictions = model.predict(X_train)
    
    # Calculate RMSE, MAE, and R²
    rmse = root_mean_squared_error(y_train, predictions)
    mae = mean_absolute_error(y_train, predictions)
    r2 = r2_score(y_train, predictions)
    
    return rmse, mae, r2

def train_and_evaluate():
    """Trains and evaluates multiple models using the pre-split train data."""
    print("[INFO] Loading training data...")
    train_df = pd.read_pickle("data/train.pkl")

    # Splitting into features and target
    X_train = train_df.drop(columns=["Temperature Sensor (°C)"])
    y_train = train_df["Temperature Sensor (°C)"]

    models = {
        "random_forest": RandomForestRegressor(),
        "xgboost": XGBRegressor(),
        "linear_regression": LinearRegression()
    }
    
    results = []
    
    for model_name, model in models.items():
        print(f"[INFO] Training {model_name}...")
        
        best_params = {}  # Default empty params

        # Perform hyperparameter tuning (except for linear regression)
        if HYPERPARAMETER_TUNING and model_name in HYPERPARAM_GRID:
            model, best_params = tune_hyperparameters(model, HYPERPARAM_GRID[model_name], X_train, y_train)
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        rmse, mae, r2 = evaluate_model(model, X_train, y_train)
        
        # Save trained model
        model_filename = f"models/{model_name}_best.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

        # Save results with hyperparameters and evaluation metrics
        results.append({
            "Model": model_name,
            "Train RMSE": rmse,
            "Train MAE": mae,
            "Train R²": r2,
            "Best Parameters": best_params if best_params else "N/A"
        })
    
    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_training_results.csv", index=False)
    print("[INFO] Model training completed. Results saved.")

if __name__ == "__main__":
    train_and_evaluate()
