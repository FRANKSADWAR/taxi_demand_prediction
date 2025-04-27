import requests
from datetime import datetime, timedelta
from typing import Optional, List
from tqdm import tqdm
from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px

def download_one_file_of_raw_data(year: int, month: int) -> Path:
    """
    This function downloads a specific month's raw taxi trip data file from a remote server and saves it 
    locally in the organization's raw data directory. 
    It returns the path to the saved file or raises an exception if the download fails.

    Args:
        year(int): the year of the data file to download
        month(int): the month of the data file to download
    Returns:
        Path: The local path top the saved file
    Raises:
        Exception if the file cannot be downloaded from the remote server
   
    """
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    response = requests.get(URL)

    if response.status_code == 200:
        path =  RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception(f'{URL} is not available')


def validate_raw_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    This function filters a DataFrame of ride records to include those with pickup_datetime
    values within the specified year and month
    Inputs

    Args:
        rides: a pandas DataFrame containing at least a pickup_datetime column.
        year: integer specifying the year to filter by.
        month: integer specifying the month to filter by.

    Returns:
        A pandas DataFrame containing only the rides with pickup_datetime within the specified year and month.


    """
    this_month_start = f'{year}-{month:02d}-01'
    next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    rides = rides[rides.pickup_datetime >= this_month_start]
    rides = rides[rides.pickup_datetime < next_month_start]

    return rides


def load_raw_data(year: int, months: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Args:
        year (int): the year of the data to load
        months: optional list of integers (1-12) specifying which months to load, if None, loads
                all months

    Returns:
        Dataframe: a pandas dataframe contaning validated ride data for the specified year and months,
                    with columns pickup_datetime ad=nd pickup_location_id
    """

    ## Initialize an empty dataframe to colect the rides
    rides = pd.DataFrame()

    ## Determines which months to process (all if months is None)
    if months is None:
        ## download only the data specified by months
        months = list(range(1,13))
    elif isinstance(months, int):
        months = [months]
    
    ## For each month, check if corresponding file exists, if not download it
    for month in months:
        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        if not local_file.exists():
            try:
                #### Download the file if it does not exist
                print(f'Downloading file {year}-{month:02d}')
                download_one_file_of_raw_data(year, month)
            except:
                print(f'{year}-{month:02d} file is not available')
                continue
        else:
            print(f'File {year}-{month:02d} is already available in storage')

        ## Load the file into Pandas
        rides_one_month = pd.read_parquet(local_file)

        ## select and rename columns
        rides_one_month = rides_one_month[['tpep_pickup_datetime','PULocationID']]
        rides_one_month.rename(columns={
            'tpep_pickup_datetime': 'pickup_datetime',
            'PULocationID': 'pickup_location_id'
        }, inplace=True)

        ## Validate the file
        rides_one_month = validate_raw_data(rides_one_month, year, month)

        ### Append to existing data
        rides = pd.concat([rides, rides_one_month])
    
    rides = rides[['pickup_datetime','pickup_location_id']]
    return rides



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


def transform_raw_data_into_ts_data(rides: pd.DataFrame) -> pd.DataFrame:
    """
    """
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('h')
    agg_rides = rides.groupby(['pickup_hour','pickup_location_id']).size().reset_index()
    agg_rides.rename(columns ={0: 'rides'},inplace=True)

    ### Add rows for(locations, pickup hours)s that had 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots



def get_cutoff_indices(data: pd.DataFrame, n_features: int, step_size: int) -> list:
    """
    
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




### FUnction that will transform all time series data into tarbular data
def transform_ts_data_into_features_and_target(ts_data: pd.DataFrame, input_sequence_len: int, step_size: int) -> pd.DataFrame:
    """
    Slices and transposes data from time series format into a features, target
    format that we can use to train a Machine Learning model
    """
    assert set(ts_data.columns) == {'pickup_hour','rides','pickup_location_id'}
    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()

    for location_id in tqdm(location_ids):
        ## keep only the time-series data for the current location in the loop
        ts_data_one_location = ts_data.loc[ts_data.pickup_location_id == location_id,['pickup_hour','rides']]
        ## pre-compute the cutoff indices to split the dataframe rows
        indices = get_cutoff_indices(ts_data_one_location,input_sequence_len,step_size)
        ## slice and transpose data into numpy arrays for features and targets
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_sequence_len), dtype = np.float32)
        y = np.ndarray(shape=(n_examples,), dtype=np.float32)
        pickup_hours = []
        
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]: idx[2]]['rides'].values
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        ## convert features from numpy to pandas
        features_one_location = pd.DataFrame(
            x, 
            columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_sequence_len))]
        )  
        features_one_location['pickup_hours'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id

        ## convert targets from numpy to pandas
        targets_one_location = pd.DataFrame(
            y,
            columns =[f'target_rides_next_hour']
        )

        ## concatenate the results
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(inplace=True,drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets['target_rides_next_hour']


def plot_rides(rides: pd.DataFrame, locations: Optional[List[int]] = None):
    """
    Plot the time-series data
    """
    rides_to_plot = rides[rides.pickup_location_id.isin(locations)] if locations else rides
    fig = px.line(rides_to_plot, x ="pickup_hour", y="rides", color = "pickup_location_id", template='none')
    fig.show()


def fetch_ride_events_from_data_warehouse(from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """
    This function is used to simulate production data by sampling historical data
    from 52 weeks ago (1 year ago).
    """
    from_date_ = from_date - timedelta(days=7*52)
    to_date_ = to_date - timedelta(days = 7 * 52)
    print(f'Fetching ride events from {from_date} to {to_date}')

    if(from_date_.year == to_date_.year) and (from_date_.month == to_date_.month):
        ## if month and year is the same download only one file
        rides = load_raw_data(year = from_date_.year, months = from_date_.month)
        rides = rides[rides.pickup_datetime >= from_date_]
        rides = rides[rides.pickup_datetime < to_date_]
    
    else:
        ## download the required files
        rides = load_raw_data(year = from_date_.year, months = from_date_.month)
        
