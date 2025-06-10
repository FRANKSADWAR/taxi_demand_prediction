import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin


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




class TemporalFeaturesEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()

        ### Generate numerical columns from datetime column
        X_["hour"] = X_["pickup_hours"].dt.hour
        X_["day_of_week"] = X_["pickup_hours"].dt.dayofweek

        X_.drop(columns = ['pickup_hours','pickup_location_id'], inplace = True)

        column_list = list(X_.columns)
        print(f"Last columns : {column_list[-5:]}")
        print(f" Shape of data after transformation: {X_.shape}")
        
        return X_
    
def get_pipeline(**hyperparams) -> Pipeline:
    ## sklearn transform
    add_feature_average_rides_last_4_weeks = FunctionTransformer(average_rides_last_4_weeks, validate = False)

    ## sklearn transform
    add_temporal_features = TemporalFeaturesEngineering()

    ## sklearn make pipeline with an estimator
    return make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )