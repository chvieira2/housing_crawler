import numpy as np
import pandas as pd
import re
import time
import requests
import os
from bs4 import BeautifulSoup
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import HardwareType, Popularity
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


def get_soup_from_url( url, sess = None, sleep_times = (1,2)):
    """
    Creates a Soup object from the HTML at the provided URL
    """
    if sess == None:
        sess = requests.session()

    # Sleeping for random seconds to avoid overload of requests
    sleeptime = np.random.uniform(sleep_times[0], sleep_times[1])
    print(f"Waiting {round(sleeptime,2)} seconds to try connecting to:\n{url}")
    time.sleep(sleeptime)

    # Load page
    user_agent_rotator = UserAgent(popularity=[Popularity.COMMON.value],
                                   hardware_types=[HardwareType.COMPUTER.value])

    HEADERS = {
            'Connection': 'keep-alive',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
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
            'referer':'https://www.google.com/'
        }
    resp = sess.get(url, headers=HEADERS)

    # Return soup object
    if resp.status_code != 200:
        return None
    return BeautifulSoup(resp.content, 'html.parser')

def crawl_ind_ad_page(url, sess=None):
    '''
    Crawl a given page for a specific add and collects the Information about Object (Angaben zum Objekt)
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
            print('Error during connection ======\n The ad does not exist.')
            return detail_dict
        else:
            try:
                soup.find('table', {'class':'table'}).find('td', {'style':'padding: 3px 8px; font-size: 1.5em; vertical-align: bottom;'})
                success = True
            except AttributeError:
                time_now = time.mktime(time.localtime())
                print(f'Sleeping until {time.strftime("%H:%M", time.localtime(time_now + 30*60))} to wait for CAPTCH to disappear....')
                time.sleep(30*60)

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

    ## Extra costs
    extra_costs = list(soup.find('table', {'class':'table'}).find_all('tr'))
    if len(extra_costs) > 0 and 'Sonstige Kosten' in str(extra_costs):
        extra_costs = [item.text.strip().replace('\n','') for item in list(extra_costs)]
        extra_costs = [item for item in extra_costs if 'Sonstige Kosten' in item]
        try:
            extra_costs = extra_costs[0].replace('Sonstige Kosten:','').strip().replace('€','')
            detail_dict['extra_costs_euros'] = np.nan if extra_costs == 'n.a.' else int(extra_costs)
        except IndexError:
            detail_dict['extra_costs_euros'] = np.nan

    ## Transfer costs
    transfer_costs = list(soup.find('table', {'class':'table'}).find_all('tr'))
    if len(transfer_costs) > 0 and 'Ablösevereinbarung' in str(transfer_costs):
        transfer_costs = [item.text.strip().replace('\n','') for item in list(transfer_costs)]
        transfer_costs = [item for item in transfer_costs if 'Ablösevereinbarung' in item]
        try:
            transfer_costs = transfer_costs[0].replace('Ablösevereinbarung:','').strip().replace('€','')
            detail_dict['transfer_costs_euros'] = np.nan if transfer_costs == 'n.a.' else int(transfer_costs)
        except IndexError:
            detail_dict['transfer_costs_euros'] = np.nan

    ## Schufa needed
    transfer_costs = list(soup.find('table', {'class':'table'}).find_all('tr'))
    if len(transfer_costs) > 0 and 'SCHUFA erwünscht' in str(transfer_costs):
        detail_dict['Schufa_needed'] = True

    ## Deposit
    kosten = soup.find('input', {'id':'kaution'})
    if kosten is not None:
        detail_dict['deposit'] = np.nan if kosten['value'] == 'n.a.' else int(kosten['value'])

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
