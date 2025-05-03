import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime

def train_test_split(
        df: pd.DataFrame,
        cutoff_date: datetime,
        target_column_name: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    
    train_data = df[df.pickup_hours < cutoff_date].reset_index(drop = True)
    test_data = df[df.pickup_hours >= cutoff_date].reset_index(drop =True)

    X_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]

    X_test = test_data.drop(columns =[target_column_name])
