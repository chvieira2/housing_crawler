
'''
The goal of this module is to collect all ads from all cities into a unified csv file.
'''

import pandas as pd
import numpy as np
import time
import re


from housing_crawler.geocoding_addresses import geocoding_address, fix_weird_address
from housing_crawler.params import dict_city_number_wggesucht
from housing_crawler.utils import save_file, get_file
from housing_crawler.string_utils import standardize_characters, capitalize_city_name, german_characters, simplify_address


def collect_cities_csvs(cities = dict_city_number_wggesucht, sleep_time_between_addresses = 5, save_after = 10):
    '''
    This function iterates through all folders of each city and saves the corresponding csvs into a single csv file.
    '''

    csvs_list = []
    for city in cities:
        city = standardize_characters(city)
        file_name = f'{city}_ads.csv'
        try:
            df_city = get_file(file_name=file_name, local_file_path=f'housing_crawler/data/{city}/Ads')
        except FileNotFoundError:
            print(f'{file_name} was not found')
            pass

            # Creates latitude and longitude columns if they don't exist
        if 'latitude' not in df_city.columns:
            df_city[['latitude', 'longitude']] = np.nan

        print(f'Geocoding addresses in {file_name}')

        # Simplifying address again is probably not necessary but address in older searchs were not simplified so I include this code here again.
        df_city['address'] = df_city['address'].apply(lambda row: simplify_address(row) if len(row.split(',')) != 3 else row) # Simplified address most have 3 strings separated by 2 commas

        total_addresses_city = len(df_city)
        counter_for_saving = 0
        for index in range(total_addresses_city):
            # Print count down
            print(f'=====> Geocoded {index+1}/{total_addresses_city}', end='\r')


            address = df_city['address'].iloc[index]
            address = fix_weird_address(address).strip().replace('  ', ' ').replace(' ,', ',')
            lat = df_city['latitude'].iloc[index]
            index_lat = list(df_city.columns).index('latitude')
            index_lon = list(df_city.columns).index('longitude')
            if pd.isnull(lat) or lat == 0:
                lat, lon = geocoding_address(address)

                # Try again with shorter address
                if pd.isnull(lat) or pd.isnull(lon):
                    # Test if there's a mispelling of lack of space in between street and house number
                    try:
                        match = re.match(r"(\D+)(\d+)(\D+)", address, re.I)
                        if match:
                            items = match.groups()
                            address = ' '.join(items).replace(' ,', ',')
                    except UnboundLocalError:
                        pass

                    try:
                        street_number = address.split(',')[0].strip().replace('  ', ' ')
                        city_name = address.split(',')[2].strip().replace('  ', ' ')
                        address_without_neigh = ', '.join([street_number, city_name]).strip().replace('  ', ' ')
                        print(f'Search did not work with "{address}". Trying with "{address_without_neigh}"')
                        time.sleep(sleep_time_between_addresses)
                        lat,lon = geocoding_address(address=address_without_neigh)
                    except IndexError:
                        print(f'Weird address format: "{address}"')
                        pass
                    # If still haven't found anything
                    if pd.isnull(lat) or pd.isnull(lon) or lat == 0 or lon == 0:
                        print('Could not find latitude and longitude.')
                        lat,lon = -1,-1
                    else:
                        print(f'Found latitude = {lat} and longitude = {lon}')

                df_city.iat[index,index_lat] = lat
                df_city.iat[index,index_lon] = lon
                time.sleep(sleep_time_between_addresses)
                counter_for_saving += 1
            else:
                pass

            # This snippet saves modifications in the csv after a number of addresses. This is so longer runs that break often can restart from where they stopped last
            if counter_for_saving >= save_after:
                save_file(df_city, file_name=file_name, local_file_path=f'housing_crawler/data/{city}/Ads')
                df_city = get_file(file_name=file_name, local_file_path=f'housing_crawler/data/{city}/Ads')
                print(f'{file_name} was updated after {counter_for_saving} addresses\n')
                counter_for_saving = 0

        print(f'Finished geocoding addresses for {capitalize_city_name(german_characters(city))}. There are {len(df_city)} ads in {file_name}. Saving modified file.')
        save_file(df_city, file_name=file_name, local_file_path=f'housing_crawler/data/{city}/Ads')

        csvs_list.append(df_city)


    all_ads_df = pd.concat(csvs_list)
    print(f'======= Csvs were collected. There are {len(all_ads_df)} ads in total. =======')

    # Get old all_Ads
    try:
        old_all_ads_df = get_file(file_name='all_encoded.csv', local_file_path=f'housing_crawler/data')
        all_ads_df = all_ads_df[all_ads_df['id'] not in old_all_ads_df['id']]
        save_file(all_ads_df, 'all_encoded.csv', local_file_path='housing_crawler/data')
    except:
        save_file(all_ads_df, 'all_encoded.csv', local_file_path='housing_crawler/data')


def long_search(day_stop_search = None, sleep_time_between_addresses = 10, start_search_from_index = 0, save_after = 10):
    '''
    This method runs the geocoding for ads until a defined date and saves results in .csv file.
    '''
    today = time.strftime(f"%d.%m.%Y", time.localtime())

    ## If no stop date is given, stops by the end of current month
    if day_stop_search is None:
        current_month = int(time.strftime(f"%m", time.localtime()))
        current_year = int(time.strftime(f"%Y", time.localtime()))
        if current_month == 12:
            next_month = 1
            next_year = current_year + 1
        else:
            next_month = current_month + 1
            next_year = current_year

        if next_month <= 9:
            day_stop_search = f'01.0{next_month}.{next_year}'
        else:
            day_stop_search = f'01.{next_month}.{next_year}'

    print(f'Geocoding will run from {today} until {day_stop_search}')

    ## Loop until stop day is reached
    while today != day_stop_search:
        today = time.strftime(f"%d.%m.%Y", time.localtime())

        # Check if between 00 and 8am, and sleep in case it is. This is because most ads are posted during the day and there's seldom need to run overnight.
        hour_of_search = int(time.strftime(f"%H", time.localtime()))
        while hour_of_search > 0 and hour_of_search < 8:
            hour_of_search = int(time.strftime(f"%H", time.localtime()))
            print(f'It is now {hour_of_search}am. Geocoding sleeping between 00 and 08am.')
            time.sleep(3600)

        # Starts the search
        cities_to_search = list(dict_city_number_wggesucht.keys())
        print(f'Starting search at {time.strftime(f"%d.%m.%Y %H:%M:%S", time.localtime())}')
        collect_cities_csvs(cities = cities_to_search[start_search_from_index:], sleep_time_between_addresses = sleep_time_between_addresses, save_after = save_after)

        # Constantly changing cities is detected by the page and goes into CAPTCH. Sleep for 15 min in between cities to avoid that.
        for temp in range(15*60)[::-1]:
            print(f'Next serch will start in {temp} seconds.', end='\r')
            time.sleep(1)
        print('\n\n\n')



if __name__ == "__main__":
     long_search()
