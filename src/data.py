import requests
from datetime import datetime, timedelta
from typing import Optional, List
from tqdm import tqdm
from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR
from pathlib import Path
import pandas as pd
import numpy as np

def download_one_file_of_raw_data(year: int, month: int) -> Path:
    """
    Gets the raw data from the web and stores it
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
    Removes the rows with pickup dates outside their valid range
    """
    this_month_start = f'{year}-{month:02d}-01'
    next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    rides = rides[rides.pickup_datetime >= this_month_start]
    rides = rides[rides.pickup_datetime < next_month_start]

    return rides


def load_raw_data(year: int, months: Optional[List[int]] = None) -> pd.DataFrame:
    """
    """
    rides = pd.DataFrame()

    if months is None:
        ## download only the data specified by months
        months = list(range(1,13))
    elif isinstance(months, int):
        months = [months]
    
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
        
