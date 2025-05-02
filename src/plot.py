import plotly.express as px
from datetime import timedelta
from typing import Optional,List
import pandas as pd


def plot_one_sample(
        features: pd.DataFrame,
        targets: pd.Series,
        example_id: int,
        predictions: Optional[pd.Series] = None,
        display_title: Optional[bool] = True
):
    features_ = features.iloc[example_id]
    if targets is not None:
        targets_ = targets.iloc[example_id]
    else:
        targets_ = None
        
    ts_columns = [c for c in features.columns if c.startswith('rides_previous_')]
    ts_values = [features_[c] for c in ts_columns] + [targets_]
    
    ts_dates = pd.date_range(features_['pickup_hour'] - timedelta(hours=len(ts_columns)),
                             features_['pickup_hour'],
                             freq='H')
    
    title = f"Pick up hour={features_['pickup_hour']}, location_id={features_['pickup_location_id']}" if display_title else None
    
    fig = px.line(
            x=ts_dates, 
            y=ts_values,
            template='plotly_dark',
            markers = True,
            title=title
    )
    
    if targets is not None:
        ## Add a green dot for the value we wish to predict
        fig.add_scatter(
            x=ts_dates[-1:],
            y=[targets_],
            line_color = 'green',
            mode='markers',
            marker_size=10,
            name='Actual values'
        )

    if predictions is not None:
        ####
        predictions_ = predictions.iloc[example_id]
        fig.add_scatter(
            x=ts_dates[-1:], y = [predictions_],
            line_color='red',
            mode='markers',
            marker_symbol='x',
            marker_size=15,
            name = 'Predictions'
        )
    return fig
    
    
    
    
def plot_ts(ts_data: pd.DataFrame, locations: Optional[List[int]] = None):
    """
    Plot the time series data
    """
    ts_data_to_plot = ts_data[ts_data.pickup_location_id.isin(locations)] if locations else ts_data

    fig = px.line(
        ts_data_to_plot,
        x="pickup_hour",
        y="rides",
        color="pickup_location_id",
        template = None
    )
    fig.show()