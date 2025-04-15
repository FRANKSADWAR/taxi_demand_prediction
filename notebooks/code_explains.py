import pandas as pd
import numpy as np
from typing import Optional, List
import plotly as px
import tqdm
import requests
from pathlib import Path




def download_one_file_of_raw_data(year: int, month: int) -> Path:
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    response = requests.get(URL)

    if response.status_code == 200:
        path = f'../data/raw/rides_{year}-{month:02d}.parquet'
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception(f'{URL} is not available')


def add_missing_slots(agg_rides: pd.DataFrame) -> pd.DataFrame:
    location_ids = agg_rides['pickup_location_id'].unique()
    full_range = pd.date_range(agg_rides['pickup_hour'].min(), agg_rides['pickup_hour'].max(), freq='h')
    output = pd.DataFrame()

    for location_id in tqdm(location_ids):
        ## keep only the rides for this location id
        agg_rides_i = agg_rides.loc[agg_rides.pickup_location_id == location_id,['pickup_hour','rides']]

        ## Quickest way to add missing values with 0 in a series
        ## Taken from https://stackoverflow.com/a/19324591
        agg_rides_i.set_index('pickup_hour',inplace = True)
        agg_rides_i.index = pd.DatetimeIndex(agg_rides_i.index)
        agg_rides_i = agg_rides_i.reindex(full_range, fill_value = 0)
        
        ## Now add back the location_id columns
        agg_rides_i['pickup_location_id'] = location_id
        output = pd.concat([output, agg_rides_i])

    output = output.reset_index().rename(columns = {'index': 'pickup_hour'})
    return output


def plot_rides(rides: pd.DataFrame, locations: Optional[List[int]] = None):
    """
    Plot the time-series data
    """
    rides_to_plot = rides[rides.pickup_location_id.isin(locations)] if locations else rides
    fig = px.line(rides_to_plot, x ="pickup_hour", y="rides", color = "pickup_location_id", template='none')
    fig.show()



def get_cutoff_indices(data: pd.DataFrame, n_features: int, step_size: int) -> list:
    """
    This function generates a list of index triplets that define sliding windows over a DataFrame, 
    useful for creating overlapping sub-sequences for time series or sequence modeling tasks
    Parameters
    -----------------------------------------------
    data: a pandas DataFrame represeneting the dataset
    n_features: integer, the length of each sub-sequence window
    step_size: integer, the number of steps to move the window forward each iteration

    Returns
    -----------------------------------------------------
    list os tuples, each containing the start, mid and end indices of a window
    """
    stop_position = len(data) -1 

    ## start the first sub-sequence at index position 0
    subseq_first_idx = 0
    subseq_mid_idx =  n_features
    subseq_last_idx = n_features + 1 
    indices = []

    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx,subseq_last_idx))

        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size
    return indices