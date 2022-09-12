
from audioop import add
import re
import time
import pandas as pd
import numpy as np
import requests
import urllib.parse

def fix_weird_address(address, weird_patterns = ['Am S Bahnhof', 'xxx', 'xx', 'Nahe', 'nahe', 'Nähe','nähe','Close To', 'Nearby','nearby', 'Close To', 'Close to', 'close to', 'close To']):
    ## Add here any other type of weird naming on addresses
    for weird in weird_patterns:
        address = address.replace(weird, '').strip().replace('  ', ' ')

    # Correcting mispelling input from users is a never ending job....
    return address.replace(' ,', ',')\
        .replace('srasse','strasse').replace('strs,','strasse,').replace('str,','strasse,').replace('Strs,','Strasse,').replace('Str,','Strasse,').replace('stasse,','strasse,').replace('Stasse,','Strasse,').replace('Strß,','Straße,').replace('strasze,','strasse,').replace('Strasze,','Strasse,')\
        .replace('Alle ', 'Allee ').replace('alle ', 'Allee ').replace('Alle,', 'Allee,').replace('alle,', 'Allee,').replace('feder','felder')\
        .replace('kungerstrasse', 'kunger strasse').replace('nummer zwei', '2')\
        .replace('Schonehauser', 'Schönhauser').replace('Warschschauer','Warschauer')\
        .replace('Dunkerstraße','Dunckerstraße').replace('Reinstraße','Rheinstraße')\
        .replace('Neltstraße', 'Neltestraße').replace('Camebridger', 'Cambridger')\
        .replace('Koperniskusstraße', 'Kopernikusstraße').replace('Düsseldoffer', 'Düsseldorfer')\
        .replace('Borndorfer','Bornsdorfer')

def geocoding_address(address, sleep_time = 900, retry=True):
    '''
    Function takes on address and returns latitude and longitude
    '''
    ## Correct weird entries in address
    address = fix_weird_address(address=address).strip().replace('  ', ' ').replace(' ,', ',')

    url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2919.83 Safari/537.36',
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

    # set lat and lon
    try:
        lat,lon = (response.json()[0]["lat"], response.json()[0]["lon"])
    except IndexError:
        lat,lon = (np.nan,np.nan)

    # If retry is True (recursivity), and lat and lon are nan, then retry code with changed address
    # Try again with shorter address
    if retry and (pd.isnull(lat) or pd.isnull(lon)):
        # Test if there's a mispelling of lack of space in between street and house number
        try:
            address_streetnumber = address.split(',')[0]
            address_other = ','.join(address.split(',')[1:])
            match = re.match(r"(\D+)(\d+)", address_streetnumber, re.I)
            if match:
                items = match.groups()
                address_streetnumber = ' '.join(items).replace(' ,', ',')
                address = ','.join([address_streetnumber,address_other])
        except UnboundLocalError:
            pass

        # Retry without zip code
        try:
            street_number = address.split(',')[0].strip().replace('  ', ' ')
            city_name = address.split(',')[2].strip().replace('  ', ' ')
            address_without_neigh = ', '.join([street_number, city_name]).strip().replace('  ', ' ')
            print(f'Search did not work with "{address}". Trying with "{address_without_neigh}"')
            time.sleep(10) # Sleep before trying again to avoid getting stuck
            lat,lon = geocoding_address(address=address_without_neigh, retry=False)
        except IndexError:
            print(f'Weird address format: "{address}"')
            pass

        # If still haven't found anything
        if pd.isnull(lat) or pd.isnull(lon):
            # Retry without street
            try:
                zip_code = address.split(',')[1].strip().replace('  ', ' ')
                city_name = address.split(',')[2].strip().replace('  ', ' ')
                address_without_street = ', '.join([zip_code, city_name]).strip().replace('  ', ' ')
                print(f'Search did not work with "{address_without_neigh}". Trying with "{address_without_street}"')
                time.sleep(3) # Sleep before trying again to avoid getting stuck
                lat,lon = geocoding_address(address=address_without_street, retry=False)
            except IndexError:
                print(f'Weird address format: "{address}"')
                pass

        # If still haven't found anything
        if pd.isnull(lat) or pd.isnull(lon):
            print('Could not find latitude and longitude.')
            lat,lon = 0,0
        else:
            print(f'Found latitude = {lat} and longitude = {lon}')

    return lat,lon





if __name__ == "__main__":
    #  df = pd.DataFrame({'address':["Am S Bahnhof Sundgauer Str , Berlin Zehlendorf",
    #                          "Müggelstraße 9, Berlin Friedrichshain",
    #                          "Brachvogelstraße 8, Berlin Kreuzberg",
    #                          'Langhansstraße 21, Berlin Prenzlauer Berg']})

    print(geocoding_address('Nazarekirchstrasse51, 13347, Berlin', sleep_time = 1, retry=True))
