#!/bin/bash

echo "======================================"
echo "Running the Machine Learning Pipeline"
echo "======================================"

# Step 1: Load the dataset
echo "--------------------------------------"
echo "[INFO] Loading the dataset..."
echo "This is the dataset subsetted based on the defined features selected."
python src/data_loader.py

# Step 2: Clean the dataset
echo "--------------------------------------"
echo "[INFO] Cleaning the dataset..."
echo "The data has been cleaned as such:"
python src/data_cleaner.py

# Step 3: Process the dataset (Feature Engineering)
echo "--------------------------------------"
echo "[INFO] Processing the dataset..."
echo "Feature engineering: binning and one-hot encoding are applied."
python src/data_processor.py

# Step 4: Split data into train and test
echo "--------------------------------------"
echo "[INFO] Splitting data into train and test splits..."
python src/data_splitter.py

# Step 5: Check if hyperparameter tuning is enabled (from config)
# Add the 'src' directory to the Python path to access config.py
HYPERPARAMETER_TUNING=$(python -c "import sys; sys.path.insert(0, 'src'); from config import HYPERPARAMETER_TUNING; print(HYPERPARAMETER_TUNING)")

if [ "$HYPERPARAMETER_TUNING" = "True" ]; then
    echo "--------------------------------------"
    echo "[INFO] Hyperparameter tuning is enabled, training models..."
    python src/model_trainer.py
    echo "--------------------------------------"
    echo "[INFO] Hyperparameter tuning is enabled, running model testing..."
    python src/model_tester.py
else
    echo "--------------------------------------"
    echo "[INFO] Hyperparameter tuning is disabled, skipping model training..."
    python src/model_tester.py  # Run model testing even if hyperparameter tuning is disabled
fi

# Final Step: Completion message
echo "--------------------------------------"
echo "[INFO] Machine learning pipeline completed successfully."
