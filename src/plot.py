import plotly.express as px
from datetime import timedelta
from typing import Optional
import pandas as pd


def plot_one_sample(
        features: pd.DataFrame,
        targets: pd.Series,
        example_id: int,
        predictions: Optional[pd.Series] = None
):
    features = features.iloc[example_id]
    targets = targets.iloc[example_id]
