# AI Singapore Technical Assessment
### Bryan Kee Tze Ren   | bryanktr@gmail.com

## Overview
This repository contains the deliverables for the AI Singapore Technical Assessment, aimed at predicting temperature conditions in a controlled farm environment for optimal plant growth. The solution includes:
1. **EDA Notebook**: A Jupyter notebook for exploratory data analysis, providing insights for model development.
2. **Machine Learning Pipeline**: Python scripts to automate data processing, model training, and evaluation. Linear Regression, Random Forest, and XGBoost were chosen for this project.

## Folder Structure
- **`data/`** : An empty data folder for the assessors to place the `agri.db` file. The folder will also store `.pkl` data generated as the pipeline runs.
- **`src/`** : Contains Python modules for each step in the machine learning pipeline.
    - **`data_loader.py`**: Loads data from the database
    - **`data_cleaner.py`**: Cleans the dataset by handling missing values and outliers.
    - **`feature_engineering.py`**: Applies binning and one-hot encoding to selected features.
    - **`data_splitter.py`**: Splits the data into training and testing sets.
    - **`model_trainer.py`**: Trains models with k-fold cross-validation and hyperparameter tuning.
    - **`model_tester.py`**: Evaluates the performance of the trained models using the test data.
- **`models/`** : Stores saved models from hyperparameter tuning. Also contains CSV files generated from training and testing results.
- **`eda.ipynb`** : Jupyter notebook for conducting exploratory data analysis.
- **`run.sh`** : A bash script to execute the entire pipeline in sequence.
- **`requirements.txt`** : Lists the required Python packages and their versions needed to run the project.
- **`README.md`** : This file, containing an overview of the project, setup instructions, and usage details.

## Instructions for Executing the Pipeline Locally on Windows
1. Clone or download the repository to your local machine.
2. Download the `agri.db` file and place it in the `data/` folder. 
3. Set Up the Python Environment:
   - Open **Git Bash**
   - Navigate to the project directory.
   - Create and activate a Python virtual environment:
     ```bash
     python -m venv venv
     source venv/Scripts/activate 
     ```
4. Install Required Packages from the `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```
5. Run the Pipeline using the `run.sh` script:
     ```bash
     bash run.sh
     ```
6. After the pipeline completes (takes around 1 minute), the trained models and evaluation results will be saved in the `models/` folder.
     - **Models**: Stored as `.pkl` files.
     - **Metrics**: Stored as `.csv` files containing training and testing performance metrics.
---

## Modifying the Config Folder
The **config.py** file provides flexibility to experiment with various settings that control data processing, feature engineering, and model training. The default values for these options are based on the EDA, but feel free to experiment with some of the options below:
- **`NEGATIVE_TO_NA`**: Set to True to convert negative values to NA
- **`IMPUTE_METHOD`**: Imputation method for missing values: "mean" , "median" , "mode" , "drop"
- **`QUANTITATIVE_VARS`**: List of quantitative variables used for modelling.
- **`CATEGORICAL_VARS`**: List of categorical variables used for modelling.
- **`EXCLUDE_VARS`**: Variables excluded from modelling
- **`BINNING_RULES`**: Modify binning rules for specific variables based on the EDA in the `eda.ipynb` file.
- **`TEST_SIZE`**: Proportion of the dataset used for the test set.
- **`HYPERPARAM_GRID`**: Grid of hyperparameters for model tuning.
- **`CV_FOLDS`**: The number of folds used for cross-validation during hyperparameter tuning.
- **`HYPERPARAMETER_TUNING`**: The default is `True` to enable hyperparameter tuning using cross-validation. If you want to avoid overriding the models that were tuned in previous runs, set this to `False`.

## Key Findings from EDA to Inform ML Pipeline
The table below summarizes the key findings from the exploratory data analysis (EDA) for each variable and the actions taken to handle them. For a detailed rationale and insights, please refer to the EDA notebook (`eda.ipynb`).

| Variable                   | Cleanliness                  | Outliers                          | Association with Temperature | Actions                                                    |
|----------------------------|------------------------------|-----------------------------------|-----------------------------|------------------------------------------------------------|
| System Location Code        | Clean                        | None                              | No                          | Excluded                                                   |
| Previous Cycle Plant Type   | Clean                        | None                              | No                          | Excluded                                                   |
| Plant Type                  | Inconsistent capitalization  | Even Distribution                 | Moderate                         | Standardized to lowercase                                                           |
| Plant Stage                 | Inconsistent capitalization  | Even Distribution                 | Moderate                         | Standardized to lowercase                                                            |
| Humidity                    | 70% Missing                  | None                              | Clustered Scatterplot           | Median Imputation, Categorical Binning                     |
| Light Intensity             | <20% Missing                 | Negative values                   | Clustered Scatterplot           | Replace negatives with NA, Median Imputation, Categorical Binning |
| Carbon Dioxide              | Clean                        | None                              | Clustered Scatterplot           | Categorical Binning                                        |
| Electrical Conductivity     | Clean                        | Negative values                   | Mild                         | Replace negatives with NA                                  |
| Oxygen                      | Clean                        | None                              | Mild                         |                                                            |
| Nitrogen                    | <20% Missing, "ppm" and "None" | None                              | Clustered Scatterplot           | Median Imputation, Categorical Binning                     |
| Phosphorus                  | <10% Missing, "ppm" and "None" | None                              | Clustered Scatterplot           | Median Imputation, Categorical Binning                     |
| Potassium                   | <10% Missing, "ppm" and "None" | None                              | Clustered Scatterplot           | Median Imputation, Categorical Binning                     |
| pH Level                    | Clean                        | None                              | Mild                         |                                                            |
| Water Level                 | <20% Missing                 | None                              | Mild                         | Median Imputation                                          |
| Temperature                 | <20% Missing                 | Negative values                   | Not Applicable               | Median Imputation, Replace negatives with NA                |

**Note**: None of the quantitative variables exhibited skew in the univariate analysis, so no transformation or scaling was applied. All categorical variables were one-hot encoded before modeling. "Clustered scatterplots" refers to distinct groupings observed in scatter plots of these variables with temperature, indicating the potential for binning or categorization.

## Description of Pipeline Flow
1. **Data Loading** [ `data_loader.py` ]
   - Loads the `agri.db` file from the `data/` folder.
   - Fetches the dataset from the `farm_data` table in the database.
   - Excludes the columns that are not associated with the response variable (Temperature).
   - Saves the subsetted data into a `.pkl` file (`data/subsetted_data.pkl`) for later use.

2. **Data Cleaning** [ `data_cleaner.py` ]
   - Standardizes categorical variables by converting them to lowercase.
   - Cleans quantitative variables by handling missing values and non-numeric characters (e.g., "None", "N/A").
   - Replaces negative values in quantitative variables with `NA` (if `NEGATIVE_TO_NA` is `True`).
   - Imputes missing values using the specified method (`mean`, `median`, `mode`, or `drop`).
   - Saves the cleaned data into a `.pkl` file (`data/cleaned_data.pkl`) for further processing.

3. **Feature Engineering** [ `data_processor.py` ]
   - Applies binning rules to quantitative variables based on the predefined rules in `BINNING_RULES`.
   - Binned columns are dropped from the dataset once new binned columns are created.
   - One-hot encodes categorical variables and binned features.
   - Saves the processed dataset into a `.pkl` file (`data/processed_data.pkl`) for use in training.

4. **Train-Test Split** [ `train_test_split.py` ]
   - Loads the processed data from the `data/processed_data.pkl` file.
   - Splits the data into training and testing sets based on the `TEST_SIZE` and `RANDOM_STATE` settings in `config.py`.
   - Saves the training and testing sets as `train.pkl` and `test.pkl` in the `data/` folder.

5. **Model Training and Evaluation** [ `model_trainer.py` ]
   - Loads the training data from the `data/train.pkl` file.
   - Defines three models: Random Forest, XGBoost, and Linear Regression.
   - Performs hyperparameter tuning using GridSearchCV for Random Forest and XGBoost
   - Trains the models and evaluates them on the training data using RMSE, MAE, and R² metrics.
   - Saves the trained models as `.pkl` files in the `models/` folder (e.g., `models/random_forest_best.pkl`).
   - Saves the model evaluation results (including RMSE, MAE, R², and hyperparameters) in a CSV file (`models/model_training_results.csv`).

6. **Model Testing** [ `model_tester.py` ]
   - Loads the test data from the `data/test.pkl` file.
   - Loads the trained models from the `models/` folder.
   - Evaluates the models on the test data using RMSE, MAE, and R² metrics.
   - Saves the test results in a CSV file (`models/model_testing_results.csv`).

## Model Evaluation

### Evaluation Metrics:
The models were evaluated based on the following metrics:
- **RMSE (Root Mean Squared Error)**: Measures the average magnitude of error between predicted and actual values. A lower value indicates better performance.
- **MAE (Mean Absolute Error)**: Represents the average absolute difference between predicted and actual values. A smaller MAE indicates a more accurate model.
- **R² (Coefficient of Determination)**: Indicates how well the model explains the variance in the target variable. A higher R² suggests a better fit.

### Model Evaluation Results:

| Model               | Test RMSE | Train RMSE | Test MAE | Train MAE | Test R²   | Train R²  |
|---------------------|-----------|------------|----------|-----------|-----------|-----------|
| **random_forest**    | 0.990     | 0.867      | 0.778    | 0.678     | 0.546     | 0.653     |
| **xgboost**          | 0.995     | 0.912      | 0.782    | 0.714     | 0.541     | 0.616     |
| **linear_regression**| 1.109     | 1.101      | 0.866    | 0.858     | 0.430     | 0.440     |

### Model Rationale and Performance:

1. **Random Forest**: Random Forest was chosen for its robustness and ability to handle non-linear relationships in the data, making it ideal for this temperature prediction task. It performed the best with the lowest RMSE and MAE, and the highest R² score on both train and test sets. However, it showed a larger difference between the training and testing results, indicating potential overfitting. Despite this, its overall performance makes it the top model for this task, especially for capturing complex patterns in the data.

2. **XGBoost**: XGBoost was selected for its efficiency and ability to handle large datasets with potentially complex interactions between features. Although it performed slightly worse than Random Forest in terms of RMSE, MAE, and R², XGBoost exhibited much less overfitting. The train and test performance were more closely aligned, suggesting that XGBoost generalizes better to unseen data. This makes it a strong contender, with good overall performance despite being marginally less accurate than Random Forest.

3. **Linear Regression**: As a baseline model, Linear Regression was included to provide a simple comparison. Its performance was the weakest, with the highest RMSE and MAE, and the lowest R². This result was expected, as Linear Regression assumes a linear relationship between the features and the target, which is a simplified assumption for this task. It serves as a benchmark model, showing the improvements offered by more complex models like Random Forest and XGBoost.

### Overall Model Performance:
All models show moderate performance with prediction errors of around ±1 degree Celsius. There is potential for improvement by expanding the hyperparameter grid, plotting feature importance for better insights, or consulting domain experts to identify any useful feature interactions that could enhance model performance. Due to time constraints, these avenues were not fully explored, but the current pipeline provides a solid foundation for further refinement.