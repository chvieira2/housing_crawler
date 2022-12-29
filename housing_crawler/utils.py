import numpy as np
import pandas as pd
import re
import time
import requests
import os
from shapely import wkt
import pickle

from shapely.geometry import Point, Polygon
import geopandas as gpd

from bs4 import BeautifulSoup
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import HardwareType, Popularity
from config.config import ROOT_DIR
from housing_crawler.params import dict_city_number_wggesucht
from housing_crawler.string_utils import standardize_characters, capitalize_city_name, german_characters, simplify_address
from housing_crawler.geocoding_addresses import geocoding_address


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

def get_soup_from_url( url, sess = None, sleep_times = (1,2)):
    """
    Creates a Soup object from the HTML at the provided URL
    """
    if sess == None:
        sess = requests.session()

    print(f"Connecting to: {url}")

    # Load page
    user_agent_rotator = UserAgent(popularity=[Popularity.COMMON.value],
                                   hardware_types=[HardwareType.COMPUTER.value])

    HEADERS = {
            'Connection': 'keep-alive',
            'Pragma': 'no-cache',
            'Cache-Control': 'max-age=0', #'no-cache',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': user_agent_rotator.get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;'
                    'q=0.9,image/webp,image/apng,*/*;q=0.8,'
                    'application/signed-exchange;v=b3;q=0.9',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-User': '?1',
            'Sec-Fetch-Dest': 'document',
            'Accept-Language': 'en-US,en;q=0.9',
        'referer':'https://www.wg-gesucht.de/'
        }

    # Start loop that only ends when get function works. This will repeat the get every 15 minutes to wait for internet to reconnect
    resp = None
    while resp is None:
        try:
            resp = sess.get(url, headers=HEADERS)
        except:
            for temp in range(15*60)[::-1]:
                print(f"There's no internet connection. Trying again in {temp} seconds.", end='\r')
                time.sleep(1)
            print('\n')

    # Return soup object
    if resp.status_code != 200:
        return None
    return BeautifulSoup(resp.content, 'html.parser')

def crawl_ind_ad_page(url, sess=None):
    '''
    Crawl a given page for a specific add and collects the information about Object (Angaben zum Objekt)
    '''

    if sess == None:
        print('Opening session')
        sess = requests.session()

    # Creates dict that will be returned at the end
    detail_dict = {}

    # Crawling page
    # Loop until CAPTCH is gone
    success = False
    while not success:
        soup = get_soup_from_url(url, sess=sess)
        if soup is None:
            print('Error during connection ======\n The ad might not exist.')
            return detail_dict
        else:
            try:
                soup.find('table', {'class':'table'}).find('td', {'style':'padding: 3px 8px; font-size: 1.5em; vertical-align: bottom;'})
                success = True
            except AttributeError:
                time_now = time.mktime(time.localtime())
                print(f'Sleeping until {time.strftime("%H:%M", time.localtime(time_now + 32*60))} to wait for CAPTCH to disappear....')
                time.sleep(32*60)

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
    mandatory_costs = list(soup.find('table', {'class':'table'}).find_all('tr'))
    if len(mandatory_costs) > 0 and 'Nebenkosten' in str(mandatory_costs):
        mandatory_costs = [item.text.strip() for item in list(mandatory_costs)]
        mandatory_costs = [item for item in mandatory_costs if 'Nebenkosten' in item]
        mandatory_costs = mandatory_costs[0].split('\n')[-1]

        try:
            mandatory_costs = mandatory_costs.replace('€','')
            detail_dict['mandatory_costs_euros'] = np.nan if mandatory_costs == 'n.a.' else int(mandatory_costs)
        except IndexError:
            detail_dict['mandatory_costs_euros'] = np.nan
    else:
        detail_dict['mandatory_costs_euros'] = 0

    ## Extra costs
    extra_costs = list(soup.find('table', {'class':'table'}).find_all('tr'))
    if len(extra_costs) > 0 and 'Sonstige Kosten' in str(extra_costs):
        extra_costs = [item.text.strip().replace('\n','') for item in list(extra_costs)]
        extra_costs = [item for item in extra_costs if 'Sonstige Kosten' in item]
        try:
            extra_costs = extra_costs[0].replace('Sonstige Kosten:','').strip().replace('€','')
            detail_dict['extra_costs_euros'] = 0 if extra_costs == 'n.a.' else int(extra_costs)
        except IndexError:
            detail_dict['extra_costs_euros'] = 0
    else:
        detail_dict['extra_costs_euros'] = 0

    ## Transfer costs
    transfer_costs = list(soup.find('table', {'class':'table'}).find_all('tr'))
    if len(transfer_costs) > 0 and 'Ablösevereinbarung' in str(transfer_costs):
        transfer_costs = [item.text.strip().replace('\n','') for item in list(transfer_costs)]
        transfer_costs = [item for item in transfer_costs if 'Ablösevereinbarung' in item]
        try:
            transfer_costs = transfer_costs[0].replace('Ablösevereinbarung:','').strip().replace('€','')
            detail_dict['transfer_costs_euros'] = 0 if transfer_costs == 'n.a.' else int(transfer_costs)
        except IndexError:
            detail_dict['transfer_costs_euros'] = 0
    else:
        detail_dict['transfer_costs_euros'] = 0

    ## Schufa needed
    info = list(soup.find('table', {'class':'table'}).find_all('tr'))
    if len(info) > 0 and 'SCHUFA erwünscht' in str(info):
        detail_dict['Schufa_needed'] = True
    else:
        detail_dict['Schufa_needed'] = False

    ## Deposit
    kosten = soup.find('input', {'id':'kaution'})
    if kosten is not None:
        detail_dict['deposit'] = np.nan if kosten['value'] == 'n.a.' else int(kosten['value'])
    else:
        detail_dict['deposit'] = 0

    ## ZIP code
    address = soup.find('div', {'class':'col-sm-4 mb10'}).find('a')

    if address is not None:
        try:
            address_zip = address.text.split('\n')[4].strip()
            foo = re.findall(r'\d+', address_zip)[0]
            address_zip = np.nan if foo in ['', ' ', None]  else int(foo)
            detail_dict['zip_code'] = address_zip
        except IndexError:
            detail_dict['zip_code'] = np.nan


    ## Look inside ad only if wg_detail exist (WGs only)
    wg_detail = soup.find_all('ul', {'class':'ul-detailed-view-datasheet print_text_left'})
    if len(wg_detail) > 1:
        die_wg = wg_detail[0]
        gesucht_wird = wg_detail[1]

        ## General details
        die_wg = [cont.text.replace('\n','').strip() for cont in die_wg.find_all('li')]

        # Home total size
        home_total_size = [item for item in die_wg if 'Wohnungsgr' in item]
        if len(home_total_size) != 0:
            home_total_size = home_total_size[0]
            home_total_size = re.sub(' +', ' ', home_total_size).replace('Wohnungsgröße: ','').replace('m²','').strip()
            detail_dict['home_total_size'] = int(home_total_size)
        else:
            detail_dict['home_total_size'] = np.nan

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
        if len(wg_type) > 3:
            selection = wg_type[3:]
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

        # Age range
        age_range = [item for item in die_wg if 'Bewohneralter:' in item]
        if len(age_range) != 0:
            age_range = age_range[0].split(':')[1]
            age_range = re.sub(' +', ' ', age_range).strip()
            detail_dict['age_range'] = age_range
        else:
            detail_dict['age_range'] = np.nan

        # Gender searched
        try:
            detail = gesucht_wird.find('li').text.replace('\n','')
            detail = re.sub(' +', ' ', detail).strip()
            detail_dict['gender_search'] = detail
            pass
        except AttributeError:
            if detail_dict.get('gender_search') is None:
                detail_dict['gender_search'] = np.nan



    ## Look for object description
    object_info = soup.find_all('div', {'class':'col-xs-6 col-sm-4 text-center print_text_left'})
    for object in object_info:
        # Energy
        try:
            detail = object.find('div', {'id':'popover-energy-certification'}).text
            if detail is not None:
                detail = object.contents[-1].replace('\n',' ').replace('m&sup2a', 'm²a')
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

    return detail_dict

def crawl_ind_ad_page2(url, sess=None):
    '''
    Crawl a given page for a specific add and collects the information about Object (Angaben zum Objekt)
    '''

    if sess == None:
        print('Opening session')
        sess = requests.session()

    # Creates dict that will be returned at the end
    detail_dict = {}

    # Crawling page
    # Loop until CAPTCH is gone
    success = False
    while not success:
        soup = get_soup_from_url(url, sess=sess)
        if soup is None:
            print('Error during connection ======\n The ad might not exist.')
            return detail_dict
        else:
            try:
                soup.find('table', {'class':'table'}).find('td', {'style':'padding: 3px 8px; font-size: 1.5em; vertical-align: bottom;'})
                success = True
            except AttributeError:
                time_now = time.mktime(time.localtime())
                print(f'Sleeping until {time.strftime("%H:%M", time.localtime(time_now + 32*60))} to wait for CAPTCH to disappear....')
                time.sleep(32*60)

    ## Check if page is not inactive
    inactivation_alert = soup.find_all("div", {"class": "alert alert-with-icon alert-warning"})
    if len(inactivation_alert) > 0:
        return 'Inactive_ad'

    ## Check if ad doesn't exist


    # Title
    title_row = soup.find('h1', {"class": "headline headline-detailed-view-title"})
    title = title_row.text.strip().replace('"','').replace('\n',' ').replace('\t',' ').replace(';','')

    # Main datails
    main = soup.find_all('div', {'class':'col-xs-6 text-center print_inline'})
    # Offer type
    if len(main) == 2:
        type_offer = 'WG'
        number_rooms = 1.0
    else:
        main = soup.find_all('div', {'class':'col-xs-4 text-center print_inline'})
        number_rooms = main[2].find('h2', {'class':'headline headline-key-facts'}).text.strip()
        type_offer = number_rooms + "-Zimmer-Wohnung"
        number_rooms = float(number_rooms.replace(',','.'))

    ## Warm rent
    price = main[1].find('h2', {'class':'headline headline-key-facts'})
    if price is not None:
        price = price.text.strip().replace('€','')
        price = np.nan if price == 'n.a.' else int(price)

    ## Size
    size = main[0].find('h2', {'class':'headline headline-key-facts'})
    if size is not None:
        size = size.text.strip().replace('m²','')
        size = np.nan if size == 'n.a.' else int(size)


    ## flatmates_list
    try:
        flatmates = soup.find("h1", {"class": "headline headline-detailed-view-title"}).find("span", {"style": "margin-right: 6px;"})['title']
        flatmates_list = [int(n) for n in re.findall('[0-9]+', flatmates)]
    except TypeError:
        flatmates_list = [0,0,0,0]

    ## Publication date and time
    # Seconds, minutes and hours are in color green (#218700), while days are in color grey (#898989)
    availability = soup.find("div", {"class": "col-sm-3"})
    try:
        published_date = availability.find("b", {"style": "color: #218700;"}).text
    except:
        published_date = availability.find("b", {"style": "color: #898989;"}).text

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

    # Offer availability dates
    availability_dates = re.findall(r'\d{2}.\d{2}.\d{4}', availability.text)


    ## Address and ZIP code
    address = soup.find('div', {'class':'col-sm-4 mb10'}).find('a')
    address = [foo.strip().replace('"','').replace('\n',' ').replace('\t',' ').replace(';','') for foo in address.text.split('\n') if foo.strip() not in ['']]
    try:
        address_zip = int(re.findall(r'\d+', address[1])[0])
    except:
        address_zip = np.nan

    city = address[1].split(' ')[1].capitalize()
    if city == 'Frankfurt': city = 'Frankfurt am Main'

    address = ', '.join([address[0], str(address_zip), city])


    ## Latitude and longitude
    lat, lon = geocoding_address(address)


    # Commercial offers from companies, often with several rooms in same building
    try:
        test_text = soup.find("div", {"class": "col-xs-8 col-sm-10"}).find("span", {"class": "label_verified"}).text
        landlord_type = test_text.replace(' ','').replace('"','').replace('\n','').replace('\t','').replace(';','')
    except AttributeError:
        landlord_type = 'Private'

    detail_dict = {
            'id': [int(url.split('.')[-2].strip())],
            'url': [str(url)],
            'type_offer': [str(type_offer)],
            'landlord_type': [str(landlord_type)],
            'title': [str(title)],
            'price_euros': [int(price)],
            'size_sqm': [int(size)],
            'available_rooms': [float(number_rooms)],
            'WG_size': [int(flatmates_list[0])],
            'available_spots_wg': [int(flatmates_list[0]-flatmates_list[1]-flatmates_list[2]-flatmates_list[3])],
            'male_flatmates': [int(flatmates_list[1])],
            'female_flatmates': [int(flatmates_list[2])],
            'diverse_flatmates': [int(flatmates_list[3])],
            'published_on': [str(published_date)],
            'published_at': [np.nan if pd.isnull(published_time) else int(published_time)],
            'address': [str(address)],
            'city': [str(german_characters(city))],
            'crawler': ['WG-Gesucht'],
            'latitude': [np.nan if pd.isnull(lat) else float(lat)],
            'longitude': [np.nan if pd.isnull(lon) else float(lon)]
        }
    if len(availability_dates) == 2:
        detail_dict['available from'] = [str(availability_dates[0])]
        detail_dict['available to'] = [str(availability_dates[1])]
    elif len(availability_dates) == 1:
        detail_dict['available from'] = [str(availability_dates[0])]
        detail_dict['available to'] = [np.nan]


    #### Further details ####
    detail_dict['details_searched'] = [True]
    ## Cold rent
    cold_rent = soup.find('table', {'class':'table'}).find('td', {'style':'padding: 3px 8px; font-size: 1.5em; vertical-align: bottom;'})
    if cold_rent is not None:
        cold_rent = cold_rent.text.strip().replace('€','')
        detail_dict['cold_rent_euros'] = [np.nan if cold_rent == 'n.a.' else int(cold_rent)]

    ## Mandatory costs
    mandatory_costs = list(soup.find('table', {'class':'table'}).find_all('tr'))
    if len(mandatory_costs) > 0 and 'Nebenkosten' in str(mandatory_costs):
        mandatory_costs = [item.text.strip() for item in list(mandatory_costs)]
        mandatory_costs = [item for item in mandatory_costs if 'Nebenkosten' in item]
        mandatory_costs = mandatory_costs[0].split('\n')[-1]

        try:
            mandatory_costs = mandatory_costs.replace('€','')
            detail_dict['mandatory_costs_euros'] = [np.nan if mandatory_costs == 'n.a.' else int(mandatory_costs)]
        except IndexError:
            detail_dict['mandatory_costs_euros'] = [np.nan]
    else:
        detail_dict['mandatory_costs_euros'] = [0]

    ## Extra costs
    extra_costs = list(soup.find('table', {'class':'table'}).find_all('tr'))
    if len(extra_costs) > 0 and 'Sonstige Kosten' in str(extra_costs):
        extra_costs = [item.text.strip().replace('\n','') for item in list(extra_costs)]
        extra_costs = [item for item in extra_costs if 'Sonstige Kosten' in item]
        try:
            extra_costs = extra_costs[0].replace('Sonstige Kosten:','').strip().replace('€','')
            detail_dict['extra_costs_euros'] = [0 if extra_costs == 'n.a.' else int(extra_costs)]
        except IndexError:
            detail_dict['extra_costs_euros'] = [0]
    else:
        detail_dict['extra_costs_euros'] = [0]

    ## Transfer costs
    transfer_costs = list(soup.find('table', {'class':'table'}).find_all('tr'))
    if len(transfer_costs) > 0 and 'Ablösevereinbarung' in str(transfer_costs):
        transfer_costs = [item.text.strip().replace('\n','') for item in list(transfer_costs)]
        transfer_costs = [item for item in transfer_costs if 'Ablösevereinbarung' in item]
        try:
            transfer_costs = transfer_costs[0].replace('Ablösevereinbarung:','').strip().replace('€','')
            detail_dict['transfer_costs_euros'] = [0 if transfer_costs == 'n.a.' else int(transfer_costs)]
        except IndexError:
            detail_dict['transfer_costs_euros'] = [0]
    else:
        detail_dict['transfer_costs_euros'] = [0]

    ## Deposit
    kosten = soup.find('input', {'id':'kaution'})
    if kosten is not None:
        detail_dict['deposit'] = [np.nan if kosten['value'] == 'n.a.' else int(kosten['value'])]
    else:
        detail_dict['deposit'] = [0]

    # zip_code
    detail_dict['zip_code'] = [address_zip]



    ## Look for object description
    object_info = soup.find_all('div', {'class':'col-xs-6 col-sm-4 text-center print_text_left'})
    for object in object_info:
        # Energy
        try:
            detail = object.find('div', {'id':'popover-energy-certification'}).text
            if detail is not None:
                detail = object.contents[-1].replace('\n',' ').replace('m&sup2a', 'm²a')
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

    ## Look inside ad only if wg_detail exist (WGs only)
    wg_detail = soup.find_all('ul', {'class':'ul-detailed-view-datasheet print_text_left'})
    if len(wg_detail) > 1:
        die_wg = wg_detail[0]
        gesucht_wird = wg_detail[1]

        ## General details
        die_wg = [cont.text.replace('\n','').strip() for cont in die_wg.find_all('li')]

        # Home total size
        home_total_size = [item for item in die_wg if 'Wohnungsgr' in item]
        if len(home_total_size) != 0:
            home_total_size = home_total_size[0]
            home_total_size = re.sub(' +', ' ', home_total_size).replace('Wohnungsgröße: ','').replace('m²','').strip()
            detail_dict['home_total_size'] = [int(home_total_size)]
        else:
            detail_dict['home_total_size'] = [np.nan]

        # Smoking
        smoking = [item for item in die_wg if 'Rauch' in item]
        if len(smoking) != 0:
            smoking = smoking[0]
            smoking = re.sub(' +', ' ', smoking).strip()
            detail_dict['smoking'] = [smoking]
        else:
            detail_dict['smoking'] = [np.nan]

        # WG type
        wg_type = [item for item in die_wg]
        if len(wg_type) > 3:
            selection = wg_type[3:]
            wg_type = [item for item in selection if 'Bewohneralte' not in item and\
                                                'Rauch' not in item and\
                                                'Sprach' not in item\
                                                ]
            if len(wg_type) > 0:
                wg_type = wg_type[0]
                wg_type = re.sub(' +', ' ', wg_type).strip()
                detail_dict['wg_type'] = [wg_type]
        else:
            detail_dict['wg_type'] = [np.nan]

        # Spoken languages
        languages = [item for item in die_wg if 'Sprach' in item]
        if len(languages) != 0:
            languages = languages[0].split(':')[1]
            languages = re.sub(' +', ' ', languages).strip()
            detail_dict['languages'] = [languages]
        else:
            detail_dict['languages'] = [np.nan]

        # Age range
        age_range = [item for item in die_wg if 'Bewohneralter:' in item]
        if len(age_range) != 0:
            age_range = age_range[0].split(':')[1]
            age_range = re.sub(' +', ' ', age_range).strip()
            detail_dict['age_range'] = [age_range]
        else:
            detail_dict['age_range'] = [np.nan]

        # Gender searched
        try:
            detail = gesucht_wird.find('li').text.replace('\n','')
            detail = re.sub(' +', ' ', detail).strip()
            detail_dict['gender_search'] = [detail]
            pass
        except AttributeError:
            if detail_dict.get('gender_search') is None:
                detail_dict['gender_search'] = [np.nan]


    ## Schufa needed
    info = list(soup.find('table', {'class':'table'}).find_all('tr'))
    if len(info) > 0 and 'SCHUFA erwünscht' in str(info):
        detail_dict['Schufa_needed'] = [True]
    else:
        detail_dict['Schufa_needed'] = [False]

    return pd.DataFrame.from_dict(detail_dict)

def lat_lon_to_polygon(df_grid):
    """Receives a dataframe with coordinates columns and adds a column of respective polygons"""
    polygons = []
    for _, row in df_grid.iterrows():
        polygons.append(Polygon([Point(row['lat_start'],row['lng_start']),
                                  Point(row['lat_start'],row['lng_end']),
                                  Point(row['lat_end'],row['lng_start']),
                                  Point(row['lat_end'],row['lng_end'])]))
    df_grid['geometry'] = polygons
    return df_grid

def get_grid_polygons_all_cities():
    try:
        df_feats = gpd.read_file(f'{ROOT_DIR}/housing_crawler/data/grid_all_cities.geojson', driver='GeoJSON')
        if 'grid_in_location' in df_feats.columns:
            df_feats = df_feats.drop(columns=['lat_start','lat_end','lng_start','lng_end','lat_center','lng_center', 'grid_in_location'])
    except:
        for city in list(dict_city_number_wggesucht.keys()):
            # Transform grids into polygons
            city_feats_df = get_file(file_name=f'FeatCount_{standardize_characters(city)}_grid_200m.csv', local_file_path=f'housing_crawler/data/{standardize_characters(city)}/WorkingTables')
            city_feats_df = city_feats_df[city_feats_df['grid_in_location']].reset_index(drop=True)
            city_feats_df = gpd.GeoDataFrame(lat_lon_to_polygon(city_feats_df), geometry='geometry', crs='wgs84')
            try:
                df_feats = pd.concat([df_feats, city_feats_df])
            except:
                df_feats = city_feats_df.copy()

        df_feats = df_feats.drop(columns=['lat_start','lat_end','lng_start','lng_end','lat_center','lng_center', 'grid_in_location'])

        df_feats.to_file(f'{ROOT_DIR}/housing_crawler/data/grid_all_cities.geojson', driver='GeoJSON')

    return gpd.GeoDataFrame(df_feats)

def standardize_features(df, features):
    df_standardized = df.copy()
    for f in features:
        mu = df[f].mean()
        sigma = df[f].std()
        df_standardized[f] = df[f].map(lambda x: (x - mu) / sigma)
    return df_standardized

def return_significative_coef(model):
    """
    Returns p_value, lower and upper bound coefficients
    from a statsmodels object.
    """
    # Extract p_values
    p_values = model.pvalues.reset_index()
    p_values.columns = ['variable', 'p_value']

    # Extract coef_int
    coef = model.params.reset_index()
    coef.columns = ['variable', 'coef']
    return p_values.merge(coef,
                          on='variable')\
                   .query("p_value<0.05").sort_values(by='coef',
                                                      ascending=False)

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def get_data(file_name_tag='ads_OSM.csv', local_file_path=f'raw_data'):
    """
    Method to get data from local environment and return a unified dataframe

    """

    csvs_list = []
    for year in ['2022','2023']:
        for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:

            file_name = f'{year}{month}_{file_name_tag}'
            local_path = f'{ROOT_DIR}/{local_file_path}/{file_name}'
            try:
                df = pd.read_csv(local_path)
                csvs_list.append(df)
            except FileNotFoundError:
                pass

    return pd.concat(csvs_list)

def obtain_latest_model():
    directory = f'{ROOT_DIR}/model/trained_models'
    pkl_files = []
    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            pkl_files.append(os.path.join(directory, file))
    pkl_files = sorted([file_name.split('/')[-1] for file_name in pkl_files])
    return pickle.load(open(f'{directory}/{pkl_files[-1]}','rb'))

def meters_to_coord(m, latitude=52.52, direction='east'):
    """
        Takes an offset in meters in a given direction (north, south, east and west) at a given latitude and returns the corresponding value in lat (north or south) or lon (east or west) degrees
        Uses the French approximation.
        More info here: https://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    import math

    # Coordinate offsets in radians
    if direction in ['x', 'east', 'west', 'e', 'w']:
        latitude = math.pi*latitude/180 # Convert latitude to raians
        return abs(m/(111_111*math.cos(latitude)))
    elif direction in ['y', 'north', 'south', 'n', 's']:
        return abs(m/111_111)
    return None


if __name__ == "__main__":

    print(crawl_ind_ad_page2('https://www.wg-gesucht.de/wg-zimmer-in-Hannover-Sudstadt.9060306.html'))
