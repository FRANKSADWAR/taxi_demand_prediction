import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgb


def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column with the average rides in the last 4 weeks i.e 7 days,
    14 days, 21 days, 28 days ago
    """
    X['average_rides_last_4_weeks'] = 0.25 * (
        X[f'rides_previous_{7*24}_hour'] + \
        X[f'rides_previous_{2*7*24}_hour'] + \
        X[f'rides_previous_{3*7*24}_hour'] + \
        X[f'rides_previous_{4*7*24}_hour']
    )

    return X

from sklearn.base import BaseEstimator, TransformerMixin

class TemporalFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        ### Generate numerical columns from datetime column
        X_["hour"] = X_["pickup_hours"].dt.hour
        X_["day_of_week"] = X_["pickup_hours"].dt.dayofweek
        
        return X_.drop(columns=['pickup_hours'])