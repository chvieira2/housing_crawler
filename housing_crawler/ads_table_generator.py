
'''
The goal of this module is to collect all ads from all cities into a unified csv file.
'''

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'. Needed to remove SettingWithCopyWarning warning when assigning new value to dataframe column
import numpy as np
import time
import re
import requests


from housing_crawler.geocoding_addresses import geocoding_address, fix_weird_address
from housing_crawler.params import dict_city_number_wggesucht
from housing_crawler.utils import save_file, get_file, crawl_ind_ad_page
from housing_crawler.string_utils import standardize_characters, capitalize_city_name, german_characters, simplify_address

def fix_older_table(df_city, file_name, city,
                      sess = None, save_after = 3):
    '''

    '''

    if sess == None:
        print('Opening session')
        sess = requests.session()


    total_rows_city = len(df_city)
    counter_for_saving = 0
    ## Loop over each row in the df
    for index_row in range(total_rows_city):

        ### Adding ad specific info
        details_searched = df_city['details_searched'].iloc[index_row]

        try:
            published_on = df_city['published_on'].iloc[index_row].split('.')
            month_published_on = int(published_on[1])
            year_published_on = int(published_on[2])
        except:
            month_published_on = 0
            year_published_on = 0

        # Check if needed
        if str(details_searched) == 'True' or str(details_searched) == 'False' or str(details_searched) == '1.0':
            pass
        # Search only from August 2022 on
        elif int(month_published_on) >= 8 and int(year_published_on) >= 2022:
            ad_url = df_city['url'].iloc[index_row]
            print(f'Collecting info for ad {df_city["id"].iloc[index_row]}', end='\n')
            ads_dict = crawl_ind_ad_page(url=ad_url, sess=sess)

            if len(list(ads_dict.keys())) == 0:
                df_city['details_searched'].iloc[index_row] = False
                pass
            ## Add ad values to main dataframe
            for key, value in ads_dict.items():
                df_city[key].iloc[index_row] = value
            counter_for_saving += 1



        ### Geocoding
        ## Geocoding is necessary because older searches did not include it. For newer searches this is redundant.

        # Simplifying address again is probably not necessary but address in older searchs were not simplified so I include this code here again.
        # df_city['address'] = df_city['address'].apply(lambda row: simplify_address(row) if len(row.split(',')) != 3 else row) # Simplified address most have 3 strings separated by 2 commas



        address = df_city['address'].iloc[index_row]
        address = fix_weird_address(address).strip().replace('  ', ' ').replace(' ,', ',')
        lat = df_city['latitude'].iloc[index_row]
        index_lat = list(df_city.columns).index('latitude')
        index_lon = list(df_city.columns).index('longitude')
        if pd.isnull(lat) or lat == np.nan or lat == -1:
            print(f'Geocoding {address}...')
            for temp in range(3)[::-1]:
                print(f'Waiting {temp} seconds before continuing.', end='\r')
                time.sleep(1)
            lat, lon = geocoding_address(address)

            df_city.iat[index_row,index_lat] = lat
            df_city.iat[index_row,index_lon] = lon

            counter_for_saving += 1

        # This snippet saves modifications in the csv after a number of addresses. This is so longer runs that break often can restart from where they stopped last
        if counter_for_saving >= save_after:
            save_file(df_city, file_name=file_name, local_file_path=f'housing_crawler/data/{city}/Ads')
            df_city = get_file(file_name=file_name, local_file_path=f'housing_crawler/data/{city}/Ads')
            print(f'{file_name} was updated after {counter_for_saving} addresses\n')
            counter_for_saving = 0

    print(f'Finished geocoding addresses for {capitalize_city_name(german_characters(city))}. There are {len(df_city)} ads in {file_name}.')
    return df_city

def collect_cities_csvs(cities = dict_city_number_wggesucht):
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

        # Update older searches with newer feautures, like geocoding of addresses or collecting ad specific info
        df_city = fix_older_table(df_city, city=city, file_name=file_name)


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

def long_search(day_stop_search = '01.01.2023', start_search_from_index = 0):
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
        while hour_of_search > 8 and hour_of_search < 8:
            hour_of_search = int(time.strftime(f"%H", time.localtime()))
            print(f'It is now {hour_of_search}am. Program sleeping between 00 and 08am.')
            time.sleep(60*60)

        # Starts the search
        cities_to_search = list(dict_city_number_wggesucht.keys())
        print(f'Starting search at {time.strftime(f"%d.%m.%Y %H:%M:%S", time.localtime())}')
        collect_cities_csvs(cities = cities_to_search[start_search_from_index:])

        # Constantly changing cities is detected by the page and goes into CAPTCH. Sleep for 15 min in between cities to avoid that.
        for temp in range(15*60)[::-1]:
            print(f'Next search will start in {temp} seconds.', end='\r')
            time.sleep(1)
        print('\n\n\n')



if __name__ == "__main__":
     long_search()
