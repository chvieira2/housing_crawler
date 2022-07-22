import numpy as np
import pandas as pd
import os
from config.config import ROOT_DIR

def create_dir(path):
    # Check whether the specified path exists or not
    if os.path.exists(path):
        pass
    else:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"The new directory has been created!")

def get_file(file_name, local_file_path='data/berlin'):
    """
    Method to get data (or a portion of it) from local environment and return a dataframe
    """
    # try:
    local_path = f'{ROOT_DIR}/{local_file_path}/{file_name}'
    df = pd.read_csv(local_path)
    print(f'===> Loaded {file_name} locally')
    return df

def save_file(df, file_name, local_file_path='housing_crawler/data/berlin'):
    # Create directory for saving
    create_dir(path = f'{ROOT_DIR}/{local_file_path}')

    # Save locally
    local_path = f'{ROOT_DIR}/{local_file_path}/{file_name}'
    df.to_csv(local_path, index=False)
    print(f"===> {file_name} saved locally")
