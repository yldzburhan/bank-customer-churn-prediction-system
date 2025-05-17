import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings

# Set random seed for reproducibility
seed = 42

# Function Transformers with error handling
def salary_rounder(x):
    """Round salary values and convert to uint64."""
    try:
        x_copy = x.copy()
        x_copy['EstimatedSalary'] = (x_copy['EstimatedSalary'] * 100).astype(np.uint64)
        return x_copy
    except Exception as e:
        warnings.warn(f"Error in salary_rounder: {str(e)}")
        return x

def age_rounder(x):
    """Round age values and convert to uint16."""
    try:
        x_copy = x.copy()
        x_copy['Age'] = (x_copy['Age'] * 10).astype(np.uint16)
        return x_copy
    except Exception as e:
        warnings.warn(f"Error in age_rounder: {str(e)}")
        return x

def balance_rounder(x):
    """Round balance values and convert to uint64."""
    try:
        x_copy = x.copy()
        x_copy['Balance'] = (x_copy['Balance'] * 100).astype(np.uint64)
        return x_copy
    except Exception as e:
        warnings.warn(f"Error in balance_rounder: {str(e)}")
        return x

def feature_generator(x):
    """Generate new features from existing ones."""
    try:
        x_copy = x.copy()
        # Safe division with handling zero division
        x_copy['IsActive_by_CreditCard'] = x_copy['HasCrCard'] * x_copy['IsActiveMember']
        x_copy['Products_Per_Tenure'] = np.where(x_copy['NumOfProducts'] != 0,
                                               x_copy['Tenure'] / x_copy['NumOfProducts'],
                                               0)
        x_copy['ZeroBalance'] = (x_copy['Balance'] == 0).astype(np.uint8)
        x_copy['AgeCat'] = np.round(x_copy.Age/20).astype(np.uint16)
        
        # Concatenate string features safely
        str_features = ['Surname', 'Geography', 'Gender', 'EstimatedSalary',
                       'CreditScore', 'Age', 'NumOfProducts', 'Tenure', 'CustomerId']
        x_copy['AllCat'] = x_copy[str_features].astype(str).agg('_'.join, axis=1)
        
        return x_copy
    except Exception as e:
        warnings.warn(f"Error in feature_generator: {str(e)}")
        return x

def svd_rounder(x):
    """Round SVD components to int64."""
    try:
        x_copy = x.copy()
        svd_cols = [col for col in x.columns if 'SVD' in col]
        for col in svd_cols:
            x_copy[col] = (x_copy[col] * 1e18).astype(np.int64)
        return x_copy
    except Exception as e:
        warnings.warn(f"Error in svd_rounder: {str(e)}")
        return x

# Create FunctionTransformers
SalaryRounder = FunctionTransformer(salary_rounder)
AgeRounder = FunctionTransformer(age_rounder)
BalanceRounder = FunctionTransformer(balance_rounder)
FeatureGenerator = FunctionTransformer(feature_generator)
SVDRounder = FunctionTransformer(svd_rounder)

class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drop specified columns from the dataset."""
    
    def __init__(self, cols):
        self.cols = cols
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        try:
            return x.drop(columns=self.cols, errors='ignore')
        except Exception as e:
            warnings.warn(f"Error in FeatureDropper: {str(e)}")
            return x

class Categorizer(BaseEstimator, TransformerMixin):
    """Convert specified columns to category dtype."""
    
    def __init__(self, cols):
        self.cols = cols
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        try:
            x_copy = x.copy()
            for col in self.cols:
                if col in x_copy.columns:
                    x_copy[col] = x_copy[col].astype('category')
            return x_copy
        except Exception as e:
            warnings.warn(f"Error in Categorizer: {str(e)}")
            return x

class Vectorizer(BaseEstimator, TransformerMixin):
    """Convert text columns to numerical features using TF-IDF and SVD."""
    
    def __init__(self, max_features=1000, cols=None, n_components=3):
        self.max_features = max_features
        self.cols = cols if cols is not None else ['Surname']
        self.n_components = n_components
        self.vectorizer_dict = {}
        self.decomposer_dict = {}
        
    def fit(self, x, y=None):
        try:
            for col in self.cols:
                if col not in x.columns:
                    warnings.warn(f"Column {col} not found in input data")
                    continue
                    
                self.vectorizer_dict[col] = TfidfVectorizer(
                    max_features=self.max_features
                ).fit(x[col].astype(str))
                
                transformed_data = self.vectorizer_dict[col].transform(x[col].astype(str))
                self.decomposer_dict[col] = TruncatedSVD(
                    random_state=seed,
                    n_components=min(self.n_components, transformed_data.shape[1])
                ).fit(transformed_data)
            
            return self
        except Exception as e:
            warnings.warn(f"Error in Vectorizer.fit: {str(e)}")
            return self
    
    def transform(self, x):
        try:
            x_copy = x.copy()
            vectorized_dfs = []
            
            for col in self.cols:
                if col not in x_copy.columns or col not in self.vectorizer_dict:
                    continue
                    
                transformed_col = self.vectorizer_dict[col].transform(x_copy[col].astype(str))
                decomposed_col = self.decomposer_dict[col].transform(transformed_col)
                
                svd_df = pd.DataFrame(
                    decomposed_col,
                    columns=[f'{col}SVD{i}' for i in range(decomposed_col.shape[1])]
                )
                vectorized_dfs.append(svd_df)
            
            if vectorized_dfs:
                vectorized_df = pd.concat(vectorized_dfs, axis=1)
                vectorized_df.index = x_copy.index
                return pd.concat([x_copy, vectorized_df], axis=1)
            
            return x_copy
        except Exception as e:
            warnings.warn(f"Error in Vectorizer.transform: {str(e)}")
            return x 