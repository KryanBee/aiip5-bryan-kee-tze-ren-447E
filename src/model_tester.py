import pandas as pd
import pickle
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def load_model(model_name):
    """Load the trained model from the pickle file."""
    model_filename = f"models/{model_name}_best.pkl"
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using the test data."""
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Calculate RMSE, MAE, and R2
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return rmse, mae, r2

def test_models():
    """Load test data and evaluate all models."""
    print("[INFO] Loading test data...")
    test_df = pd.read_pickle("data/test.pkl")
    
    # Splitting into features and target
    X_test = test_df.drop(columns=["Temperature Sensor (°C)"])
    y_test = test_df["Temperature Sensor (°C)"]
    
    # Models to test
    models = ["random_forest", "xgboost", "linear_regression"]
    results = []
    
    # Test each model
    for model_name in models:
        print(f"[INFO] Testing {model_name} model...")
        
        model = load_model(model_name)  # Load the model
        
        # Evaluate model
        rmse, mae, r2 = evaluate_model(model, X_test, y_test)
        
        # Save the results
        results.append({
            "Model": model_name,
            "Test RMSE": rmse,
            "Test MAE": mae,
            "Test R2": r2
        })
    
    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_testing_results.csv", index=False)
    print("[INFO] Model testing completed. Results saved.")

if __name__ == "__main__":
    test_models()
