import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LogScaleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols):

        self.numeric_cols = numeric_cols
        self.mean_ = {}
        self.std_ = {}

    def fit(self, X, y=None):
        X_temp = X.copy()
        
        for col in self.numeric_cols:
            # 1. Log transform (log1p handles 0 values safely)
            x_col_log = np.log1p(X_temp[col].values)
            
            # 2. Compute mean and std for standard scaling
            self.mean_[col] = x_col_log.mean()
            self.std_[col]  = x_col_log.std()
        
        return self
    
    def transform(self, X, y=None):
        X_trans = X.copy()
        
        for col in self.numeric_cols:
            # 1. Log transform
            x_col_log = np.log1p(X_trans[col].values)
            
            # 2. Standard scale using learned mean_ and std_
            mean_val = self.mean_[col]
            # Avoid dividing by zero if std_ is extremely small
            std_val  = self.std_[col] if self.std_[col] != 0 else 1e-12
            
            x_col_scaled = (x_col_log - mean_val) / std_val
            
            # 3. Assign transformed data back
            X_trans[col] = x_col_scaled
        
        return X_trans
    



