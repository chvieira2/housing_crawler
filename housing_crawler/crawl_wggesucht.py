# from cmath import isnan
import re
import time
import datetime
from dateutil.relativedelta import relativedelta
import requests
# import json
# import logging
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from housing_crawler.abstract_crawler import Crawler
from housing_crawler.string_utils import remove_prefix, simplify_address, standardize_characters, capitalize_city_name, german_characters
from housing_crawler.utils import save_file, get_file, crawl_ind_ad_page
from housing_crawler.params import dict_city_number_wggesucht
from housing_crawler.geocoding_addresses import geocoding_address

from config.config import ROOT_DIR



class CrawlWgGesucht(Crawler):
    """Implementation of Crawler interface for WgGesucht"""

    # __log__ = logging.getLogger('housing_crawler')

    def __init__(self):
        self.base_url = 'https://www.wg-gesucht.de/'

        # logging.getLogger("requests").setLevel(logging.WARNING)
        self.existing_findings = []

        # The number after the name of the city is specific to the city
        self.dict_city_number_wggesucht = dict_city_number_wggesucht

    def url_builder(self, location_name, page_number,
                    filters):
        # Make sure that the city name is correct
        location_name_for_url = capitalize_city_name(standardize_characters(location_name, separator=' ')).replace(' ', '-')

        filter_code = []

        if "wg-zimmer" in filters:
            filter_code.append('0')
        if "1-zimmer-wohnungen" in filters:
            filter_code.append('1')
        if "wohnungen" in filters:
            filter_code.append('2')
        if "haeuser" in filters:
            filter_code.append('3')

        filter_code = '+'.join(filter_code)
        filter_code = ''.join(['.']+[filter_code]+['.1.'])

        return self.base_url +\
            "-und-".join(filters) +\
                '-in-'+ location_name_for_url + '.' + self.dict_city_number_wggesucht.get(location_name) +\
                    filter_code + str(page_number) + '.html'

    def get_soup_from_url(self, url, sess = None):
        """
        Creates a Soup object from the HTML at the provided URL

        Overwrites the method inherited from abstract_crawler. This is
        necessary as we need to reload the page once for all filters to
        be applied correctly on wg-gesucht.
        """
        if sess == None:
            print('Opening session')
            sess = requests.session()

        # Setup an agent
        # print('Rotating agent')
        self.rotate_user_agent()

        # Second page load
        print(f"Connecting to:\n{url}")
        time.sleep(3)

        # Start loop that only ends when get function works. This will repeat the get every 15 minutes to wait for internet to reconnect
        resp = None
        while resp is None:
            try:
                resp = sess.get(url, headers=self.HEADERS)
            except:
                for temp in range(15*60)[::-1]:
                    print(f"There's no internet connection. Trying again in {temp} seconds.", end='\r')
                    time.sleep(1)
                print('\n')

        # print(f"Got response code {resp.status_code}")

        # Return soup object
        if resp.status_code != 200:
            return None
        return BeautifulSoup(resp.content, 'html.parser')

    def extract_data(self, soup):
        """Extracts all exposes from a provided Soup object"""

        # Find ads
        findings = soup.find_all(lambda e: e.has_attr('id') and e['id'].startswith('liste-'))
        findings_list = [
        e for e in findings if e.has_attr('class') and not 'display-none' in e['class']
        ]
        print(f"Extracted {len(findings_list)} entries")
        return findings_list

    def request_soup(self,url, sess):
        '''
        Extract findings with requests library
        '''
        if sess == None:
            print('Opening session')
            sess = requests.session()

        soup = self.get_soup_from_url(url, sess=sess)
        if soup is None:
            return None
        return self.extract_data(soup)

    def parse_urls(self, location_name, page_number, filters, sess, sleep_time = 32*60):
        """Parse through all exposes in self.existing_findings to return a formated dataframe.
        """
        # Process city name to match url
        location_name = capitalize_city_name(location_name)

        print(f'Processed location name. Searching for page number {page_number+1} for {location_name}')

        # Create list with all urls for crawling
        list_urls = [self.url_builder(location_name = location_name, page_number = page_number,
                    filters = filters)]

        print(f'Created url for crawling')

        # Crawling each page and adding findings to self.new_findings list
        for url in list_urls:
            if sleep_time>0:
                # Loop until CAPTCH is gone
                success = False
                while not success:
                    new_findings = self.request_soup(url, sess=sess)
                    if new_findings is None:
                        print('Probably error 500')
                        self.rotate_user_agent()
                        time_now = time.mktime(time.localtime())
                        print(f'Sleeping until {time.strftime("%H:%M", time.localtime(time_now + sleep_time))} to wait for page to allow connection again.')
                        time.sleep(sleep_time)
                    elif len(new_findings) == 0:
                        time_now = time.mktime(time.localtime())
                        print(f'Sleeping until {time.strftime("%H:%M", time.localtime(time_now + sleep_time))} to wait for CAPTCH to disappear....')
                        time.sleep(sleep_time)
                    else:
                        success = True
            else:
                new_findings = self.request_soup(url)

            if len(new_findings) == 0:
                print('====== Stopped retrieving pages. Probably stuck at CAPTCH ======')
                break

            self.existing_findings = self.existing_findings + (new_findings)

    def update_database(self, df, location_name, df_previous_version = []):
        ## First obtain older ads for updating table
        if len(df_previous_version) == 0:
            try:
                df_previous_version = get_file(file_name=f'''{datetime.date.today().strftime('%Y%m')}_{location_name}_ads.csv''',
                                local_file_path=f'housing_crawler/data/{location_name}/Ads')
                # print('Obtained older ads')
            except FileNotFoundError:
                df_previous_version = pd.DataFrame(columns = df.columns)
                print('No older ads found for this month. Creating brand new csv.')

        # Exclude weird ads without an id number from newer ads
        df = df[df['id'] != 0]

        # Make sure new df and old df have same columns
        for new_column in df_previous_version.columns:
            if new_column not in df.columns:
                df[new_column] = np.nan
        for new_column in df.columns:
            if new_column not in df_previous_version.columns:
                df_previous_version[new_column] = np.nan

        # Add new ads to older list and discard copies. Keep = last is important because wg-gesucht keeps on refreshing the post date of some ads (private paid ads?). With keep = last I keep the very first entry only
        # This snippet is probably unnecessary as duplicates aren't searched anymore and is left over from before I fixed the bug when duplicates were still searched
        new_df = pd.concat([df,df_previous_version]).drop_duplicates(subset='id', keep="last").reset_index(drop=True)



        print(f'''{len(new_df)-len(df_previous_version)} new ads were added to the list.\n
                    There are now {len(new_df)} ads in {datetime.date.today().strftime('%Y%m')}_{location_name}_ads.csv.''')
        print('\n')

        # Save updated list
        save_file(df = new_df, file_name=f'''{datetime.date.today().strftime('%Y%m')}_{location_name}_ads.csv''',
                  local_file_path=f'housing_crawler/data/{location_name}/Ads')
        return len(new_df)-len(df_previous_version)

    def crawl_all_pages(self, location_name, number_pages,
                    filters = ["wg-zimmer","1-zimmer-wohnungen","wohnungen","haeuser"],
                    path_save = None, sess=None):
        '''
        Main crawling function. Function will first connect to all pages and save findings (ads) using the parse_url method. Next, it obtain older ads table to which newer ads will be added.
        '''
        if path_save is None:
            path_save = f'housing_crawler/data/{standardize_characters(location_name)}/Ads'

        if sess == None:
            print('Opening session')
            sess = requests.session()

        # First page load to set filters; response is discarded
        print('Making first call to set filters')
        url_set_filters = self.url_builder(location_name = location_name, page_number = 1,
                    filters = filters)

        # Start loop that only ends when get function works. This will repeat the get every 15 minutes to wait for internet to reconnect
        resp = None
        while resp is None:
            try:
                resp = sess.get(url_set_filters, headers=self.HEADERS)
            except:
                for temp in range(15*60)[::-1]:
                    print(f"There's no internet connection. Trying again in {temp} seconds.", end='\r')
                    time.sleep(1)
                print('\n')

        # Rotate agent
        print('Rotating agent...')
        self.rotate_user_agent()
        resp = None
        while resp is None:
            try:
                resp = sess.get(url_set_filters, headers=self.HEADERS)
            except:
                for temp in range(15*60)[::-1]:
                    print(f"There's no internet connection. Trying again in {temp} seconds.", end='\r')
                    time.sleep(1)
                print('\n')

        # Obtain older ads, or create empty table if there are no older ads.
        try:
            previous_3_months_df = pd.concat([
                get_file(file_name=f'''{(datetime.date.today() - relativedelta(months=1)).strftime('%Y%m')}_{standardize_characters(location_name)}_ads.csv''',
                            local_file_path=f'''housing_crawler/data/{standardize_characters(location_name)}/Ads'''),
                get_file(file_name=f'''{(datetime.date.today() - relativedelta(months=2)).strftime('%Y%m')}_{standardize_characters(location_name)}_ads.csv''',
                            local_file_path=f'''housing_crawler/data/{standardize_characters(location_name)}/Ads'''),
                get_file(file_name=f'''{(datetime.date.today() - relativedelta(months=3)).strftime('%Y%m')}_{standardize_characters(location_name)}_ads.csv''',
                            local_file_path=f'''housing_crawler/data/{standardize_characters(location_name)}/Ads'''),
            ])
        except FileNotFoundError:
            previous_3_months_df=pd.DataFrame({'url':[]})

        zero_new_ads_in_a_row = 0
        total_added_findings = 0
        for page_number in range(number_pages):
            # Obtaining pages
            # No need for time.sleep here as there's already one before calling BeautifulSoup
            self.parse_urls(location_name = location_name, page_number= page_number,
                            filters = filters, sess=sess)

            # This snippet needs to run inside the page_number loop so it's constantly updated with previous ads in the same month
            try:
                current_month_df = get_file(file_name=f'''{datetime.date.today().strftime('%Y%m')}_{standardize_characters(location_name)}_ads.csv''',
                                local_file_path=f'''housing_crawler/data/{standardize_characters(location_name)}/Ads''')
                print(f'Loaded this months {standardize_characters(location_name)}_ads.csv with {len(current_month_df)} ads in total')
            except FileNotFoundError:
                current_month_df=pd.DataFrame({'url':[]})

            old_df = pd.concat([current_month_df,previous_3_months_df], axis = 0)


            # Extracting info of interest from pages
            entries = []
            total_findings = len(self.existing_findings)
            for index in range(total_findings):
                # Print count down
                print(f'=====> Parsed {index+1}/{total_findings}', end='\r')
                row = self.existing_findings[index]

                # Ad title and url
                title_row = row.find('h3', {"class": "truncate_title"})
                title = title_row.text.strip().replace('"','').replace('\n',' ').replace('\t',' ').replace(';','')
                ad_url = self.base_url + remove_prefix(title_row.find('a')['href'], "/")

                ad_id = ad_url.split('.')[-2].strip()


                # Save time by not parsing old ads
                # To check if add is old, check if the url/id already exist in the old_df table
                if ad_id == '':
                    pass
                elif (ad_url in list(old_df['url'])) or ('asset_id' in ad_url) or (ad_id in list(old_df['id']) or (int(ad_id) in list(old_df['id']))):
                    pass
                else:
                    # Add sleep time to avoid multiple sequencial searches in short time that would be detected by the site
                    # No sleep between searches is needed because wg-gesucht CAPTCH seems to be triggered by total number of accesses
                    for temp in range(10)[::-1]:
                        print(f'Waiting {temp} seconds before continuing.', end='\r')
                        time.sleep(1)
                    print('\n')
                    ## Get ad specific details
                    ad_details = crawl_ind_ad_page(url=ad_url,sess=sess)

                    # Commercial offers from companies, often with several rooms in same building
                    try:
                        test_text = row.find("div", {"class": "col-xs-9"})\
                    .find("span", {"class": "label_verified ml5"}).text
                        landlord_type = test_text.replace(' ','').replace('"','').replace('\n','').replace('\t','').replace(';','')
                    except AttributeError:
                        landlord_type = 'Private'

                    ## Room details and address
                    detail_string = row.find("div", {"class": "col-xs-11"}).text.strip().split("|")
                    details_array = list(map(lambda s: re.sub(' +', ' ',
                                                            re.sub(r'\W', ' ', s.strip())),
                                            detail_string))

                    # Offer type
                    type_offer = details_array[0]

                    # Number of rooms
                    rooms_tmp = re.findall(r"^\d*[., ]?\d*", details_array[0])[0] # Finds int or dec in beginning of word. Needed to deal with '2.5', '2,5' or '2 5' sized flats
                    rooms_tmp = float(rooms_tmp.replace(',','.').replace(' ','.'))
                    if 'WG' in type_offer:
                        type_offer = 'WG'
                        rooms = 1
                    else:
                        rooms = rooms_tmp if rooms_tmp>0 else 0

                    ## Address
                    address = details_array[2].replace('"','').replace('\n',' ').replace('\t',' ').replace(';','')\
                        + ', ' + details_array[1].replace('"','').replace('\n',' ').replace('\t',' ').replace(';','')

                    address = simplify_address(address)
                    address = re.sub(' +', ' ', address) # Remove multiple spaces

                    # Add the zip code info in place of neighbourhood name
                    try:
                        zip_code = ad_details['zip_code']
                    except KeyError:
                        zip_code = np.nan
                    if isinstance(zip_code, int):
                        address = address.split(', ')
                        address[1] = str(zip_code)
                        address = ', '.join(address)


                    ## Latitude and longitude
                    for temp in range(2)[::-1]:
                        print(f'Geocoding starts in {temp} seconds.', end='\r')
                        time.sleep(1)
                    lat, lon = geocoding_address(address)

                    # Flatmates
                    try:
                        flatmates = row.find("div", {"class": "col-xs-11"}).find("span", {"class": "noprint"})['title']
                        flatmates_list = [int(n) for n in re.findall('[0-9]+', flatmates)]
                    except TypeError:
                        flatmates_list = [0,0,0,0]

                    ### Price, size and date
                    numbers_row = row.find("div", {"class": "middle"})

                    # Price
                    price = numbers_row.find("div", {"class": "col-xs-3"}).text.strip().split(' ')[0]
                    # Prevent n.a. entries in price
                    try:
                        int(price)
                    except ValueError:
                        price = 0

                    # Offer availability dates
                    availability_dates = re.findall(r'\d{2}.\d{2}.\d{4}',
                                    numbers_row.find("div", {"class": "text-center"}).text)

                    # Size
                    size = re.findall(r'\d{1,4}\sm²',
                                    numbers_row.find("div", {"class": "text-right"}).text)
                    if len(size) == 0:
                        size = ['0']

                    size = re.findall('^\d+', size[0])[0]

                    ## Publication date and time
                    # Seconds, minutes and hours are in color green (#218700), while days are in color grey (#898989)
                    try:
                        published_date = row.find("div", {"class": "col-xs-9"})\
                    .find("span", {"style": "color: #218700;"}).text
                    except:
                        published_date = row.find("div", {"class": "col-xs-9"})\
                    .find("span", {"style": "color: #898989;"}).text

                    # For ads published mins or secs ago, publication time is the time of the search
                    hour_of_search = int(time.strftime(f"%H", time.localtime()))
                    if 'Minut' in published_date or 'Sekund' in published_date:
                        published_date = time.strftime("%d.%m.%Y", time.localtime())
                        published_time = hour_of_search

                    # For ads published hours ago, publication time is the time of search minus the hours difference. That might lead to negative time of the day and that's corrected below. That could also lead to publication date of 00, and that's also corrected below.
                    elif 'Stund' in published_date:
                        hour_diff = int(re.findall('[0-9]+', published_date)[0])
                        published_time = hour_of_search - hour_diff
                        if published_time < 0:
                            # Fix publication hour
                            published_time = 24 + published_time
                            # Fix publication date to the day before the day of the search
                            published_date = time.strftime("%d.%m.%Y", time.localtime(time.mktime(time.localtime())-24*60*60))

                        else:
                            published_date = time.strftime("%d.%m.%Y", time.localtime())

                    # For ads published days ago, publication time is NaN
                    elif 'Tag' in published_date:
                        day_diff = int(re.findall('[0-9]+', published_date)[0])
                        published_date = time.strftime("%d.%m.%Y", time.localtime(time.mktime(time.localtime())-day_diff*24*60*60))
                        published_time = np.nan

                    # For ads published at specified date (ads older than 5 days), publication time is NaN
                    else:
                        published_date = published_date.split(' ')[1]
                        published_time = np.nan



                    ### Create dataframe with info
                    details = {
                        'id': int(ad_id),
                        'url': str(ad_url),
                        'type_offer': str(type_offer),
                        'landlord_type': str(landlord_type),
                        'title': str(title),
                        'price_euros': int(price),
                        'size_sqm': int(size),
                        'available_rooms': float(rooms),
                        'WG_size': int(flatmates_list[0]),
                        'available_spots_wg': int(flatmates_list[0]-flatmates_list[1]-flatmates_list[2]-flatmates_list[3]),
                        'male_flatmates': int(flatmates_list[1]),
                        'female_flatmates': int(flatmates_list[2]),
                        'diverse_flatmates': int(flatmates_list[3]),
                        'published_on': str(published_date),
                        'published_at': np.nan if pd.isnull(published_time) else int(published_time),
                        'address': str(address),
                        'city': str(german_characters(location_name)),
                        'crawler': 'WG-Gesucht',
                        'latitude': np.nan if pd.isnull(lat) else float(lat),
                        'longitude': np.nan if pd.isnull(lon) else float(lon)
                    }
                    if len(availability_dates) == 2:
                        details['available from'] = str(availability_dates[0])
                        details['available to'] = str(availability_dates[1])
                    elif len(availability_dates) == 1:
                        details['available from'] = str(availability_dates[0])
                        details['available to'] = np.nan

                    ## Add ad details to the dictionary
                    details.update(ad_details)

                    entries.append(details)


            # Reset existing_findings
            self.existing_findings = []

            # Create the dataframe
            df = pd.DataFrame(entries)

            if len(df)>0:
                # Save info as df in csv format
                n_ads_added = self.update_database(df=df, location_name=standardize_characters(location_name), df_previous_version=current_month_df)
            else:
                n_ads_added = 0
                print('===== No new entries were found. =====')

            total_added_findings += n_ads_added

            if int(n_ads_added) == 0:
                zero_new_ads_in_a_row += 1
                print(f'{zero_new_ads_in_a_row} pages with no new ads added in a series')
            else:
                zero_new_ads_in_a_row = 0


            if zero_new_ads_in_a_row >=3:
                break

        print('\n')
        print(f'''========= {total_added_findings} ads in total were added to {datetime.date.today().strftime('%Y%m')}_{location_name}_ads.csv. =========''')
        print('\n')

    def long_search(self, day_stop_search = '01.01.2024', pages_per_search = 100, start_search_from_index = 0):
        '''
        This method runs the search for ads until a defined date and saves results in .csv file.
        '''

        print('Opening session')
        sess = requests.session()

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

        print(f'Search will run from {today} until {day_stop_search}')

        ## Loop until stop day is reached
        while today != day_stop_search:
            today = time.strftime(f"%d.%m.%Y", time.localtime())

            # Check if between 00 and 8am, and sleep in case it is. This is because most ads are posted during the day and there's seldom need to run overnight.
            hour_of_search = int(time.strftime(f"%H", time.localtime()))
            while hour_of_search > 8 and hour_of_search < 8:
                hour_of_search = int(time.strftime(f"%H", time.localtime()))
                for temp in range(60*60)[::-1]:
                    print(f'It is now {hour_of_search}am. Program sleeping between 00 and 08am. Waiting {temp} seconds before continuing.', end='\r')
                    time.sleep(1)

            # Starts the search
            cities_to_search = list(dict_city_number_wggesucht.keys())#[::-1]
            for city in cities_to_search[start_search_from_index:]:
                print(f'Starting search at {time.strftime(f"%d.%m.%Y %H:%M:%S", time.localtime())}')
                self.crawl_all_pages(location_name = city, number_pages = pages_per_search,
                            filters = ["wg-zimmer","1-zimmer-wohnungen","wohnungen","haeuser"],
                            sess = sess)

            # Constantly searching is detected by the page and goes into CAPTCH. Sleep for 5 min in between cities to avoid that.
            for temp in range(5*60)[::-1]:
                print(f'Next search will start in {temp} seconds.', end='\r')
                time.sleep(1)
            print('\n\n\n')



if __name__ == "__main__":
    while True:
        try:
            CrawlWgGesucht().long_search()
        except ConnectionError:
            for temp in range(15*60)[::-1]:
                print('============================================================')
                print("There's no internet connection. Trying again in {temp} seconds.")
                print('============================================================')
                time.sleep(1)

    # print(datetime.date.today().strftime('%Y%m'))
