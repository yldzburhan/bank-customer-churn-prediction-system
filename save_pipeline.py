import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from category_encoders import CatBoostEncoder as CECatBoostEncoder, MEstimateEncoder 
from xgboost import XGBClassifier

try:
    from custom_transformers import (
        SalaryRounder,      # FunctionTransformer(salary_rounder_func)
        AgeRounder,         # FunctionTransformer(age_rounder_func)
        FeatureGenerator,   # FunctionTransformer(feature_generator_func)
        Vectorizer          # Custom Vectorizer class
    )
    print("Custom transformers imported successfully.")
except ImportError:
    print("ERROR: 'custom_transformers.py' file not found or does not contain the necessary classes/functions.")
    print("Please ensure this file is correctly configured.")
    exit()
except Exception as e:
    print(f"An error occurred while importing custom transformers: {e}")
    exit()


# Random seed for reproducibility
seed = 42

# Loading Data
print("Loading data...")
try:
    df_train_comp = pd.read_csv('data/train.csv', index_col='id').astype({
        'IsActiveMember': np.uint8,
        'HasCrCard': np.uint8
    })

    df_orig_train = pd.read_csv('data/churn_modeling.csv', index_col='RowNumber')
    
    if 'IsActiveMember' in df_orig_train.columns:
        df_orig_train['IsActiveMember'] = df_orig_train['IsActiveMember'].astype(np.uint8)
    if 'HasCrCard' in df_orig_train.columns:
        df_orig_train['HasCrCard'] = df_orig_train['HasCrCard'].astype(np.uint8)

    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Data file not found. Please ensure the 'data' folder and CSV files within it are in the correct location.")
    print(f"Error detail: {e}")
    exit()
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    exit()

print("Preparing training data...")

try:
    X_combined = pd.concat([df_orig_train.drop('Exited', axis=1, errors='ignore'), 
                            df_train_comp.drop('Exited', axis=1, errors='ignore')], 
                           ignore_index=True)
    y_combined = pd.concat([df_orig_train['Exited'], 
                            df_train_comp['Exited']], 
                           ignore_index=True)
    print(f"Combined training data size: X_combined shape {X_combined.shape}, y_combined shape {y_combined.shape}")
    print("Training data prepared successfully.")
except Exception as e:
    print(f"An error occurred during data merging or preparation: {e}")
    exit()


# Best XGBoost Parameters
xgb_params = {
    'eta': 0.04007938900538817, 
    'max_depth': 5, 
    'subsample': 0.8858539721226424, 
    'colsample_bytree': 0.41689519430449395, 
    'min_child_weight': 0.4225662401139526, 
    'reg_lambda': 1.7610231110037127, 
    'reg_alpha': 1.993860687732973,
    'random_state': seed,
    'tree_method': 'hist',
    'n_estimators': 1000
}

# Creating Pipeline
print("Creating pipeline...")
pipeline = make_pipeline(
    SalaryRounder,      # from custom_transformers (FunctionTransformer)
    AgeRounder,         # from custom_transformers (FunctionTransformer)
    FeatureGenerator,   # from custom_transformers (FunctionTransformer)
    Vectorizer(         # from custom_transformers (Custom Class)
        cols=['Surname', 'AllCat', 'EstimatedSalary', 'CustomerId'], 
        max_features=1000, 
        n_components=3 
    ),
    CECatBoostEncoder(  # from category_encoders
        cols=['CustomerId', 'Surname', 'EstimatedSalary', 'AllCat', 'CreditScore']
    ),
    MEstimateEncoder(cols=['Geography', 'Gender']), # from category_encoders
    XGBClassifier(**xgb_params)
)
print("Pipeline created successfully.")

# Training the Pipeline
print("Training pipeline... This may take some time.")
try:
    pipeline.fit(X_combined, y_combined)
    print("Pipeline trained successfully.")
except Exception as e:
    print(f"An error occurred while training the pipeline: {e}")
    print("Please ensure that features, transformers, and model parameters are correctly configured.")
    exit()

# Saving the Pipeline
output_filename = 'churn_model_pipeline.joblib'
print(f"Saving pipeline as '{output_filename}'...")
try:
    joblib.dump(pipeline, output_filename)
    print(f"Pipeline saved successfully as '{output_filename}'!") 
except Exception as e:
    print(f"An error occurred while saving the pipeline: {e}")
