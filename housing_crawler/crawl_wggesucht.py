from cmath import isnan
import re
import time
import requests
import json
# import logging
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from housing_crawler.abstract_crawler import Crawler
from housing_crawler.string_utils import remove_prefix, simplify_address, standardize_characters, capitalize_city_name, german_characters
from housing_crawler.utils import save_file, get_file
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

    def get_soup_from_url(self, url, sess = None, sleep_times = (1,2)):
        """
        Creates a Soup object from the HTML at the provided URL

        Overwrites the method inherited from abstract_crawler. This is
        necessary as we need to reload the page once for all filters to
        be applied correctly on wg-gesucht.
        """
        if sess == None:
            print('Opening session')
            sess = requests.session()

        # Sleeping for random seconds to avoid overload of requests
        sleeptime = np.random.uniform(sleep_times[0], sleep_times[1])
        print(f"Waiting {round(sleeptime,2)} seconds to try connecting to:\n{url}")
        time.sleep(sleeptime)

        # Setup an agent
        print('Rotating agent')
        self.rotate_user_agent()

        # Second page load
        print(f"Connecting to page...")
        resp = sess.get(url, headers=self.HEADERS)
        print(f"Got response code {resp.status_code}")

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

    def parse_urls(self, location_name, page_number, filters, sess, sleep_time = 1800):
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
                        pass
                    elif len(new_findings) == 0:
                        time_now = time.mktime(time.localtime())
                        print(f'Sleeping until {time.strftime("%H:%M", time.localtime(time_now + sleep_time))} to wait for CAPTCH to disappear....')
                        time.sleep(sleep_time)
                        sleep_time += sleep_time
                    else:
                        success = True
            else:
                new_findings = self.request_soup(url)

            if len(new_findings) == 0:
                print('====== Stopped retrieving pages. Probably stuck at CAPTCH ======')
                break

            self.existing_findings = self.existing_findings + (new_findings)

    def save_df(self, df, location_name):
        ## First obtain older ads for updating table
        try:
            old_df = get_file(file_name=f'{location_name}_ads.csv',
                            local_file_path=f'housing_crawler/data/{location_name}/Ads')
            print('Obtained older ads')
        except:
            old_df = pd.DataFrame(columns = df.columns)
            print('No older ads found')

        # Exclude weird ads without an id number
        df = df[df['id'] != 0]

        # Make sure new df and old df have same columns
        for new_column in old_df.columns:
            if new_column not in df.columns:
                df[new_column] = np.nan

        # Add new ads to older list and discard copies. Keep = first is important because wg-gesucht keeps on refreshing the post date of some ads (private paid ads?). With keep = first I keep the first entry
        df = pd.concat([df,old_df]).drop_duplicates(subset='id', keep="last").reset_index(drop=True)
        print(f'''{len(df)-len(old_df)} new ads were added to the list.\n
                    There are now {len(df)} ads in {location_name}_ads.csv.''')

        # Save updated list
        save_file(df = df, file_name=f'{location_name}_ads.csv',
                  local_file_path=f'housing_crawler/data/{location_name}/Ads')
        return len(df)-len(old_df)

    def crawl_ind_ad_page(self, url, sess=None):
        '''
        Crawl a given page for a specific add and collects the Information about Object (Angaben zum Objekt)
        '''

        if sess == None:
            print('Opening session')
            sess = requests.session()

        # Crawling page
        # Loop until CAPTCH is gone
        success = False
        while not success:
            soup = self.get_soup_from_url(url, sess=sess)
            if soup is None:
                print('Error during connection ======')
                return None
            elif len(soup) == 0:
                time_now = time.mktime(time.localtime())
                print(f'Sleeping until {time.strftime("%H:%M", time.localtime(time_now + 30*60))} to wait for CAPTCH to disappear....')
                time.sleep(30*60)
            else:
                success = True

        # Creates dict that will be returned at the end
        detail_dict = {}

        ## Check if page is not inactive
        inactivation_alert = soup.find_all("div", {"class": "alert alert-with-icon alert-warning"})
        if len(inactivation_alert) > 0:
            detail_dict['details_searched'] = False
            return detail_dict
        detail_dict['details_searched'] = True


        ## Cold rent
        cold_rent = soup.find('table', {'class':'table'}).find('td', {'style':'padding: 3px 8px; font-size: 1.5em; vertical-align: bottom;'})
        if cold_rent is not None:
            cold_rent = cold_rent.text.strip().replace('€','')
            detail_dict['cold_rent_euros'] = np.nan if cold_rent == 'n.a.' else int(cold_rent)

        ## Mandatory costs
        mandatory_costs = list(soup.find('table', {'class':'table'}).find_all('td'))
        if len(mandatory_costs) > 0:
            try:
                mandatory_costs = mandatory_costs[3].text.strip().replace('€','')
                detail_dict['mandatory_costs_euros'] = np.nan if mandatory_costs == 'n.a.' else int(mandatory_costs)
            except IndexError:
                detail_dict['mandatory_costs_euros'] = np.nan

        ## Extra costs
        extra_costs = list(soup.find('table', {'class':'table'}).find_all('td'))
        if len(extra_costs) > 0:
            try:
                extra_costs = extra_costs[5].text.strip().replace('€','')
                detail_dict['extra_costs_euros'] = np.nan if extra_costs == 'n.a.' else int(extra_costs)
            except IndexError:
                detail_dict['extra_costs_euros'] = np.nan

        ## Transfer costs
        transfer_costs = list(soup.find('table', {'class':'table'}).find_all('td'))
        if len(transfer_costs) > 0 and 'Ablösevereinbarung' in str(transfer_costs):
            try:
                transfer_costs = transfer_costs[9].text.strip().replace('€','')
                detail_dict['transfer_costs_euros'] = np.nan if transfer_costs == 'n.a.' else int(transfer_costs)
            except IndexError:
                detail_dict['transfer_costs_euros'] = np.nan

        ## Deposit
        kosten = soup.find('input', {'id':'kaution'})
        if kosten is not None:
            detail_dict['deposit'] = np.nan if kosten['value'] == 'n.a.' else int(kosten['value'])

        ## ZIP code
        address = soup.find('div', {'class':'col-sm-4 mb10'}).find('a')

        if address is not None:
            address_zip = address.text.split('\n')[4].strip()
            foo = re.findall(r'\d+', address_zip)[0]
            address_zip = np.nan if foo in ['', ' ', None]  else int(foo)
            detail_dict['zip_code'] = address_zip




        ## Look inside ad only if wg_detail exist (WGs only)
        wg_detail = soup.find_all('ul', {'class':'ul-detailed-view-datasheet print_text_left'})

        if len(wg_detail) > 0:
            die_wg = wg_detail[0]
            gesucht_wird = wg_detail[1]


            ## General details
            die_wg = [cont.text.replace('\n','').strip() for cont in die_wg.find_all('li')]

            # Smoking
            smoking = [item for item in die_wg if 'Rauch' in item]
            if len(smoking) != 0:
                smoking = smoking[0]
                smoking = re.sub(' +', ' ', smoking).strip()
                detail_dict['smoking'] = smoking
            else:
                detail_dict['smoking'] = np.nan

            # WG type
            wg_type = [item for item in die_wg]
            if len(wg_type) > 4:
                selection = wg_type[4:]
                wg_type = [item for item in selection if 'Bewohneralte' not in item and\
                                                    'Rauch' not in item and\
                                                    'Sprach' not in item\
                                                    ]
                if len(wg_type) > 0:
                    wg_type = wg_type[0]
                    wg_type = re.sub(' +', ' ', wg_type).strip()
                    detail_dict['wg_type'] = wg_type
            else:
                detail_dict['wg_type'] = np.nan

            # Spoken languages
            languages = [item for item in die_wg if 'Sprach' in item]
            if len(languages) != 0:
                languages = languages[0].split(':')[1]
                languages = re.sub(' +', ' ', languages).strip()
                detail_dict['languages'] = languages
            else:
                detail_dict['languages'] = np.nan

            # Spoken languages
            age_range = [item for item in die_wg if 'Bewohneralter:' in item]
            if len(age_range) != 0:
                age_range = age_range[0].split(':')[1]
                age_range = re.sub(' +', ' ', age_range).strip()
                detail_dict['age_range'] = age_range
            else:
                detail_dict['age_range'] = np.nan


            # Looking for gender
            try:
                detail = gesucht_wird.find('li').text.replace('\n','')
                detail = re.sub(' +', ' ', detail).strip()
                detail_dict['gender_search'] = detail
                pass
            except AttributeError:
                if detail_dict.get('gender_search') is None:
                    detail_dict['gender_search'] = np.nan





        ## Look for object description
        all_info = soup.find_all('div', {'class':'col-xs-6 col-sm-4 text-center print_text_left'})
        for object in all_info:
            # Energy
            try:
                detail = object.find('div', {'id':'popover-energy-certification'}).text
                if detail is not None:
                    detail = object.contents[-1]
                    detail = re.sub(' +', ' ', detail)
                    detail_dict['energy'] = detail.strip()
                    pass
            except AttributeError:
                if detail_dict.get('energy') is None:
                    detail_dict['energy'] = np.nan

            # WG possible
            try:
                detail = object.find('span', {'class':'mdi mdi-account-group mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['wg_possible'] = detail
                    pass
            except AttributeError:
                if detail_dict.get('wg_possible') is None:
                    detail_dict['wg_possible'] = np.nan

            # Building type
            try:
                detail = object.find('span', {'class':'mdi mdi-city mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['building_type'] = detail
                    pass
            except AttributeError:
                if detail_dict.get('building_type') is None:
                    detail_dict['building_type'] = np.nan

            # Building floor
            try:
                detail = object.find('span', {'class':'mdi mdi-office-building mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['building_floor'] = detail
                    pass
            except AttributeError:
                if detail_dict.get('building_floor') is None:
                    detail_dict['building_floor'] = np.nan

            # Furniture
            try:
                detail = object.find('span', {'class':'mdi mdi-bed-double-outline mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['furniture'] = detail
                    pass
            except AttributeError:
                if detail_dict.get('furniture') is None:
                    detail_dict['furniture'] = np.nan

            # Kitchen
            try:
                detail = object.find('span', {'class':'mdi mdi-silverware-fork-knife mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['kitchen'] = detail
                    pass
            except AttributeError:
                if detail_dict.get('kitchen') is None:
                    detail_dict['kitchen'] = np.nan

            # Shower type
            try:
                detail = object.find('span', {'class':'mdi mdi-shower mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['shower_type'] = detail
                    pass
            except AttributeError:
                if detail_dict.get('shower_type') is None:
                    detail_dict['shower_type'] = np.nan

            # TV
            try:
                detail = object.find('span', {'class':'mdi mdi-monitor mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['TV'] = detail
                    pass
            except AttributeError:
                if detail_dict.get('TV') is None:
                    detail_dict['TV'] = np.nan

            # Floor type
            try:
                detail = object.find('span', {'class':'mdi mdi-layers mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['floor_type'] = detail
                    pass
            except AttributeError:
                if detail_dict.get('floor_type') is None:
                    detail_dict['floor_type'] = np.nan

            # Heating
            try:
                detail = object.find('span', {'class':'mdi mdi-fire mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['heating'] = detail
                    pass
            except AttributeError:
                if detail_dict.get('heating') is None:
                    detail_dict['heating'] = np.nan

            # Public transport distance
            try:
                detail = object.find('span', {'class':'mdi mdi-bus mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['public_transport_distance'] = detail
                    pass
            except AttributeError:

                if detail_dict.get('public_transport_distance') is None:
                    detail_dict['public_transport_distance'] = np.nan

            # Internet
            try:
                detail = object.find('span', {'class':'mdi mdi-wifi mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['internet'] = detail
                    pass
            except AttributeError:

                if detail_dict.get('internet') is None:
                    detail_dict['internet'] = np.nan

            # Parking
            try:
                detail = object.find('span', {'class':'mdi mdi-car mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['parking'] = detail
                    pass
            except AttributeError:
                if detail_dict.get('parking') is None:
                    detail_dict['parking'] = np.nan

            # Extras
            try:
                detail = object.find('span', {'class':'mdi mdi-folder mdi-36px noprint'}).text
                if detail is not None:
                    detail = object.text.replace('\n','')
                    detail = re.sub(' +', ' ', detail).strip()
                    detail_dict['extras'] = detail
                    pass
            except AttributeError:
                if detail_dict.get('extras') is None:
                    detail_dict['extras'] = np.nan


        print(detail_dict)

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
        sess.get(url_set_filters, headers=self.HEADERS)

        zero_new_ads_in_a_row = 0
        total_added_findings = 0
        for page_number in range(number_pages):
            # Obtaining pages
            self.parse_urls(location_name = location_name, page_number= page_number,
                        filters = filters, sess=sess)


            # Obtain older ads, or create empty table if there are no older ads. Needs to run inside the page_number loop so it's constantly updated with previous ads
            try:
                old_df = get_file(file_name=f'{standardize_characters(location_name)}_ads.csv',
                                local_file_path=f'housing_crawler/data/{standardize_characters(location_name)}/Ads')
            except FileNotFoundError:
                old_df=pd.DataFrame({'url':[]})


            # Extracting info of interest from pages
            print(f"Crawling {len(self.existing_findings)} ads")
            entries = []
            total_findings = len(self.existing_findings)
            for index in range(total_findings):
                # Print count down
                print(f'=====> Geocoded {index+1}/{total_findings}', end='\r')
                row = self.existing_findings[index]

                ### Commercial offers from companies, often with several rooms in same building
                try:
                    test_text = row.find("div", {"class": "col-xs-9"})\
                .find("span", {"class": "label_verified ml5"}).text
                    landlord_type = test_text.replace(' ','').replace('"','').replace('\n','').replace('\t','').replace(';','')
                except AttributeError:
                    landlord_type = 'Private'

                # Ad title and url
                title_row = row.find('h3', {"class": "truncate_title"})
                title = title_row.text.strip().replace('"','').replace('\n',' ').replace('\t',' ')\
                    .replace(';','')
                ad_url = self.base_url + remove_prefix(title_row.find('a')['href'], "/")

                # Save time by not parsing old ads
                # To check if add is old, check if the url already exist in the table
                if ad_url in list(old_df['url']):
                    # print('dasdasdadsasdasdasdasd')
                    # time.sleep(50)
                    pass
                else:
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

                    # Address
                    address = details_array[2].replace('"','').replace('\n',' ').replace('\t',' ').replace(';','')\
                        + ', ' + details_array[1].replace('"','').replace('\n',' ').replace('\t',' ').replace(';','')

                    address = simplify_address(address)

                    ## Latitude and longitude
                    lat, lon = geocoding_address(address)
                    # Add sleep time to avoid multiple sequencial searches in short time that would be detected by the site
                    time.sleep(3)

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
                        'id': int(ad_url.split('.')[-2]),
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

                    entries.append(details)

            # Reset existing_findings
            self.existing_findings = []

            # Create the dataframe
            df = pd.DataFrame(entries)

            if len(df)>0:
                # Save info as df in csv format
                ads_added = self.save_df(df=df, location_name=standardize_characters(location_name))
            else:
                ads_added = 0
                print('===== No new entries were found. =====')

            total_added_findings += ads_added

            if int(ads_added) == 0:
                    zero_new_ads_in_a_row += 1
                    print(f'{zero_new_ads_in_a_row} pages with no new ads added in a series')
            else:
                zero_new_ads_in_a_row = 0


            if zero_new_ads_in_a_row >=3:
                break

        print(f'========= {total_added_findings} ads in total were added to {location_name}_ads.csv =========')

    def long_search(self, day_stop_search = '01.01.2023', pages_per_search = 25, start_search_from_index = 0):
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
            while hour_of_search > 0 and hour_of_search < 8:
                hour_of_search = int(time.strftime(f"%H", time.localtime()))
                print(f'It is now {hour_of_search}am. Program sleeping between 00 and 08am.')
                time.sleep(3600)

            # Starts the search
            cities_to_search = list(dict_city_number_wggesucht.keys())
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
    CrawlWgGesucht().long_search()
    # CrawlWgGesucht().crawl_all_pages('Stuttgart', 100)

    # df = get_file(file_name=f'berlin_ads.csv',
    #                         local_file_path=f'housing_crawler/data/berlin/Ads')

    # CrawlWgGesucht().save_df(df, 'berlin')

    # CrawlWgGesucht().crawl_ind_ad_page(url = 'https://www.wg-gesucht.de/wg-zimmer-in-Stuttgart-Sud.9397478.html')


# https://www.wg-gesucht.de/wohnungen-in-Berlin-Prenzlauer-Berg.8740207.html
# https://www.wg-gesucht.de/wg-zimmer-in-Stuttgart-Degerloch.9468004.html
# https://www.wg-gesucht.de/wohnungen-in-Stuttgart-Sud.9522935.html
# https://www.wg-gesucht.de/1-zimmer-wohnungen-in-Stuttgart-Ost.9518976.html
# https://www.wg-gesucht.de/wohnungen-in-Stuttgart-Ost.9523598.html
# https://www.wg-gesucht.de/wohnungen-in-Stuttgart-Feuerbach.9522133.html
# https://www.wg-gesucht.de/wg-zimmer-in-Stuttgart-Moehringen.9519926.html
# https://www.wg-gesucht.de/wg-zimmer-in-Stuttgart-Sud.9397664.html
# https://www.wg-gesucht.de/wg-zimmer-in-Stuttgart-Sud.9397478.html
