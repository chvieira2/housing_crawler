
import re
import time
import pandas as pd
import numpy as np
import requests
import urllib.parse
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

def fix_weird_address(address, weird_patterns = ['Am S Bahnhof ', 'xxx', 'xx', 'nahe', 'Nähe']):
    ## Add here any other type of weird naming on addresses
    for weird in weird_patterns:
        address = address.replace(weird, '')

    return address.replace('kungerstrasse', 'kunger strasse').replace('nummer zwei', '2').replace('srasse','strasse')

def geocoding_df(df, column = 'address'):



    # Fixes weird addresses before Geocoding
    df = df[column].apply(lambda row: fix_weird_address(address=row))

    locator = Nominatim(user_agent='myGeocoder')
    # 1 - convenient function to delay between geocoding calls
    geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
    # 2- - create location column
    df['location'] = df[column].apply(geocode)
    # 3 - create longitude, laatitude and altitude from location column (returns tuple)
    df['location'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None).tolist()
    # 4 - Remove entries without detectable location
    # df = df.dropna(subset=['location'])
    # 5 - split point column into latitude, longitude and altitude columns
    try:
        df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df['location'].tolist(), index=df.index)
    except ValueError:
        df[['latitude', 'longitude', 'altitude']] = [np.nan,np.nan,np.nan]
    # 6 - Remove unnecessary columns
    return df.drop(columns=['location', 'altitude'])

def geocoding_address(address, sleep_time = 900):
    '''
    Function takes on address and returns latitude and longitude
    '''
    ## Coorect weird entries in address
    address = fix_weird_address(address=address)

    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2919.83 Safari/537.36'
    }

    # Loop until request is successfull
    success = False
    while not success:
        try:
            response = requests.get(url, headers=HEADERS)
        except requests.exceptions.ConnectionError:
            print(f'Got requests.exceptions.ConnectionError')
            time_now = time.mktime(time.localtime())
            print(f'Sleeping for {sleep_time/60} min until {time.strftime("%H:%M", time.localtime(time_now + sleep_time))} to wait for API to be available again.')
            time.sleep(sleep_time)
            sleep_time += sleep_time
            pass

        if response.status_code != 200:
            print(f'Got response code {response.status_code}')
            time_now = time.mktime(time.localtime())
            print(f'Sleeping for {sleep_time/60} min until {time.strftime("%H:%M", time.localtime(time_now + sleep_time))} to wait for API to be available again.')
            time.sleep(sleep_time)
            sleep_time += sleep_time
        else:
            success = True
    try:
        return (response.json()[0]["lat"], response.json()[0]["lon"])

    except IndexError:
        return (np.nan,np.nan)




if __name__ == "__main__":
    #  df = pd.DataFrame({'address':["Am S Bahnhof Sundgauer Str , Berlin Zehlendorf",
    #                          "Müggelstraße 9, Berlin Friedrichshain",
    #                          "Brachvogelstraße 8, Berlin Kreuzberg",
    #                          'Langhansstraße 21, Berlin Prenzlauer Berg']})

    print(geocoding_address(address = "Südendstraße , Schöneberg, Berlin"))
