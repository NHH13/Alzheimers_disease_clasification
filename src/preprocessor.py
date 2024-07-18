import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FlexiblePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features):
        self.scaler = StandardScaler()
        self.selected_features = selected_features
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, X, y=None):
        X = self._preprocess(X)
        self.scaler.fit(X[self.selected_features])
        self.is_fitted = True
        return self

    def transform(self, X):
        X = self._preprocess(X)
        if not self.is_fitted:
            return X[self.selected_features]
        X[self.selected_features] = self.scaler.transform(X[self.selected_features])
        return X[self.selected_features]

    def _preprocess(self, X):
        X = X.copy()
        
        if 'AgeGroup' not in X.columns and 'Age' in X.columns:
            X['AgeGroup'] = pd.cut(X['Age'], bins=[0, 50, 60, 70, 80, 100], 
                                   labels=['0-50', '51-60', '61-70', '71-80', '81-100'])
            X['AgeGroup'] = self.label_encoder.fit_transform(X['AgeGroup'])
        
        if 'HealthRisk' not in X.columns:
            risk_columns = ['CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension']
            X['HealthRisk'] = X[risk_columns].sum(axis=1)
        
        columns_to_drop = ['Age', 'DoctorInCharge', 'PatientID']
        X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])
        
        return X

def apply_transformations(df):
    df = df.copy()
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 50, 60, 70, 80, 100], labels=['0-50', '51-60', '61-70', '71-80', '81-100'], ordered=True)
    df = df.drop(columns=['Age'])
    df['HealthRisk'] = df[['CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension']].sum(axis=1)
    
    # Apply Label Encoding to AgeGroup
    label_encoder = LabelEncoder()
    df['AgeGroup'] = label_encoder.fit_transform(df['AgeGroup'])
    
    return df
