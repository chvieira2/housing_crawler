import re
import time
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'. Needed to remove SettingWithCopyWarning warning when assigning new value to dataframe column
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px

import statsmodels.formula.api as smf
import scipy.stats as stats

from housing_crawler.utils import save_file, get_file, crawl_ind_ad_page


def prepare_data(ads_df):
    '''
    This function fixes minor mistakes in the table and correctly sets the data types
    '''
    ## Minor mistakes
    ads_df = ads_df.rename(columns={'WG_size':'capacity',
                                            'available from':'available_from',
                                            'available to':'available_to',
                                            'Schufa_needed':'schufa_needed',
                                             'TV':'tv'})

    ads_df['landlord_type'] = ads_df['landlord_type'].replace('s','Verifiziert').replace('VerifiziertesUnternehmen','Verifiziert')


    ## Preapare data types
    ads_df['published_at'] = ads_df['published_at'].astype('Int64') # Int64 can take NaN while int or int64 won't
    ads_df['published_on'] = pd.to_datetime(ads_df['published_on'], format = "%d.%m.%Y")
    # ads_df['published_at'] = pd.to_datetime(ads_df['published_at'], format = "%H")
    ads_df['available_from'] = pd.to_datetime(ads_df['available_from'], format = "%d.%m.%Y")
    ads_df['available_to'] = pd.to_datetime(ads_df['available_to'], format = "%d.%m.%Y")


    ## details_searched
    # indicates if details have been searched or not.
    # Over time this is going to become useless but it is relevant as ~50.000 flats were searched before the code for searching deatils have been implemented
    ads_df['details_searched'] = ads_df['details_searched'].replace('1.0',1).replace('False',0).replace('True',1).replace(np.nan,0).astype('int64')



    ## type_offer
    # Simplify type of offer to match searches at wg-gesucht.de
    ads_df['type_offer_simple'] = ['Single-room flat' if ('1 Zimmer Wohnung' in offer_type or '1 Zimmer Wohnung Wohnungen' in offer_type) else offer_type for offer_type in list(ads_df['type_offer'])]
    ads_df['type_offer_simple'] = ['Apartment' if ('Zimmer Wohnung' in offer_type) else offer_type for offer_type in list(ads_df['type_offer_simple'])]
    ads_df['type_offer_simple'] = ['WG' if ('WG' in offer_type) else offer_type for offer_type in list(ads_df['type_offer_simple'])]
    ads_df['type_offer_simple'] = ['House' if ('Haus' in offer_type) else offer_type for offer_type in list(ads_df['type_offer_simple'])]
    ads_df = ads_df.drop(columns=['type_offer'])

    return ads_df

def filter_out_bad_entries(ads_df, country = 'Germany',
                           price_max = 4000, price_min = 50,
                          size_max = 400, size_min = 3,
                          date_max = None, date_min = None, date_format = "%d.%m.%Y"):

    try:
        # Filter ads in between desired dates. Standard is to use ads from previous 3 months
        if date_max == None or date_max == 'today':
            date_max = pd.to_datetime(time.strftime(date_format, time.localtime()), format = date_format)
        elif isinstance(date_max,str):
            date_max = pd.to_datetime(date_max, format = date_format)

        if date_min == None:
            date_min = datetime.date.today() + relativedelta(months=-3)
            date_min = pd.to_datetime(date_min.strftime(date_format), format = date_format)
        elif isinstance(date_min,str):
            date_min = pd.to_datetime(date_min, format = date_format)

        ads_df['temp_col'] = ads_df['published_on'].apply(lambda x: x >= date_min and x <= date_max)

        ads_df = ads_df[ads_df['temp_col']].drop(columns=['temp_col'])
    except ValueError:
        print('Date format was wrong. Please input a date in the format 31.12.2020 (day.month.year), or specify the date format you want to use using the "date_format" option.')


    ## Filter out unrealistic offers
    ads_df = ads_df.query(f'price_euros <= {price_max}\
                         & price_euros > {price_min}\
                         & size_sqm <= {size_max}\
                         & size_sqm >= {size_min}')

    if country.lower() in ['germany', 'de']:
        # Germany bounding box coordinates from here: https://gist.github.com/graydon/11198540
        ads_df['latitude'] = [lat if (lat>47.3024876979 and lat<54.983104153) else np.nan for lat in list(ads_df['latitude'])]
        ads_df['longitude'] = [lon if (lon>5.98865807458 and lon<15.0169958839) else np.nan for lon in list(ads_df['longitude'])]

    ads_df.reset_index(inplace=True, drop = True)

    return ads_df

def transform_columns_into_numerical(ads_df):

    ## wg_possible
    # Only relevant for houses and flats
    # 1 = allowed to turn into WG
    # 0 = not allowed to turn into WG (no response)
    # NaN = not searched for details (see details_searched)

    ads_df['wg_possible'] = [0 if item != item else 1 for item in ads_df['wg_possible']] # np.nan doesn't equals itself
    ads_df.loc[ads_df['details_searched'] == 0, 'wg_possible'] = np.nan
    ads_df.loc[ads_df['type_offer_simple'] == 'WG', 'wg_possible'] = 1.0


    ## building_floor
    # indicates the level from the ground. Ground level is 0.
    # Ambiguous values were given fractional definitions ('Hochparterre':0.5, 'Tiefparterre':-0.5).
    # 6 indicates values above 5, not necessarily the 6th floor
    # NaN = indicates lack of response or not searched for details (see details_searched)
    mapping_dict = {'EG':0, '1. OG':1, '2. OG':2, '3. OG':3, '4. OG':4, '5. OG':5, 'höher als 5. OG':6,
                    'Hochparterre':0.5, 'Dachgeschoss':2, 'Tiefparterre':-0.5, 'Keller':-1}
    ads_df['building_floor']= ads_df['building_floor'].map(mapping_dict)


    ## furniture
    # 1 = möbliert
    # 0.5 = teilmöbliert
    # 0 = no answer (assumed to be not furnitured)
    # NaN = not searched for details (see details_searched)
    mapping_dict = {'möbliert':1, 'teilmöbliert':0.5, 'möbliert, teilmöbliert':0.5}
    ads_df['furniture']= ads_df['furniture'].map(mapping_dict).replace(np.nan,0)
    ads_df.loc[ads_df['details_searched'] == 0, 'furniture'] = np.nan


    ## kitchen
    # 1 = kitchen ('Eigene Küche' or 'Einbauküche')
    # 0.75 = 'Kochnische' (room + kitchen)
    # 0.5 = 'Küchenmitbenutzung' (shared kitchen)
    # 0 = no kitchen (Nicht vorhanden [not available] or no response)
    # NaN = not searched for details (see details_searched)
    mapping_dict = {'Nicht vorhanden':0, 'Küchenmitbenutzung':0.5, 'Kochnische':0.75, 'Eigene Küche':1, 'Einbauküche':1}
    ads_df['kitchen']= ads_df['kitchen'].map(mapping_dict).replace(np.nan,0)
    ads_df.loc[ads_df['details_searched'] == 0, 'kitchen'] = np.nan


    ## public_transport_distance
    # Distance in minutes to public transportation
    # NaN = indicates lack of response or not searched for details (see details_searched)
    ads_df['public_transport_distance'] = ads_df['public_transport_distance'].apply(lambda x: np.nan if x!=x else int(x.split(' Min')[0]))


    ## extras
    # 1 = yes
    # 0 = no answer (assumed to not exist)
    # NaN = not searched for details (see details_searched)
    unique_extras = ['Waschmaschine', 'Spülmaschine', 'Terrasse', 'Balkon', 'Garten', 'Gartenmitbenutzung', 'Keller', 'Aufzug',
                     'Haustiere', 'Fahrradkeller', 'Dachboden']

    for option in unique_extras:
        option_name = 'extras_' + option.lower().replace('ü','ue')
        ads_df[option_name] = np.nan
        ads_df.loc[ads_df['details_searched'] == 1.0, option_name] = 0.0
        ads_df.loc[[option in item if item==item else False for item in ads_df['extras'] ], option_name] = 1
    ads_df = ads_df.drop(columns=['extras'])


    ## languages
    # 1 = yes
    # 0 = no answer (assumed to not exist)
    # NaN = not searched for details (see details_searched)
    unique_languages = ['Deutsch', 'Englisch']

    for option in unique_languages:
        option_name = 'languages_' + option.lower()
        ads_df[option_name] = np.nan
        ads_df.loc[ads_df['details_searched'] == 1.0, option_name] = 0.0
        ads_df.loc[[option in item if item==item else False for item in ads_df['languages']], option_name] = 1

    ## Number of languages
    # NaN = no answer or not searched for details (see details_searched)
    ads_df['number_languages'] = ads_df['languages'].apply(lambda x: len(str(x).split(',')) if x == x else np.nan)


    ads_df = ads_df.drop(columns=['languages'])


    ## smoking
    # 1 = allowed everywhere
    # 0.75 = allowed in room
    # 0.5 = allowed in the balcony (outside)
    # 0 = not allowed or no response
    # NaN = not searched for details (see details_searched)
    mapping_dict = {'Rauchen nicht erwünscht':0, 'Rauchen auf dem Balkon erlaubt':0.5, 'Rauchen im Zimmer erlaubt':0.75, 'Rauchen überall erlaubt':1}
    ads_df['smoking']= ads_df['smoking'].map(mapping_dict).replace(np.nan,0)
    ads_df.loc[ads_df['details_searched'] == 0, 'smoking'] = np.nan


    ## Age range
    # min age range
    ads_df['min_age_flatmates'] = np.nan
    ads_df['min_age_flatmates'] = [np.nan if item != item else \
                                   np.nan if str(item).startswith('bis') else \
                               re.findall('[0-9]+', item)[0] \
                               for item in list(ads_df['age_range'])]

    # max age range
    ads_df['max_age_flatmates'] = np.nan
    ads_df['max_age_flatmates'] = [np.nan if item != item else \
                                   np.nan if str(item).startswith('ab') else \
                                   re.findall('[0-9]+', item)[0] if str(item).startswith('bis') else \
                               re.findall('[0-9]+', item)[1] \
                               for item in list(ads_df['age_range'])]

    return ads_df

def hot_encode_columns(ads_df):


    ## wg_type
    # 1 = yes
    # 0 = no answer (assumed to not exist)
    # NaN = not searched for details (see details_searched)
    unique_wg_type = ['Studenten-WG', 'keine Zweck-WG', 'Männer-WG', 'Business-WG', 'Wohnheim', 'Vegetarisch/Vegan',
                   'Alleinerziehende', 'funktionale WG', 'Berufstätigen-WG', 'gemischte WG', 'WG mit Kindern',
                   'Verbindung', 'LGBTQIA+', 'Senioren-WG', 'inklusive WG', 'WG-Neugründung']

    for option in unique_wg_type:
        option_name = 'wg_type_' + option.lower().replace('-wg','').replace(' wg','').replace('wg ','')\
        .replace('ä','ae').replace(' ','_').replace('/','_').replace('-','_').replace('+','')
        ads_df[option_name] = np.nan
        ads_df.loc[ads_df['details_searched'] == 1.0, option_name] = 0.0
        ads_df.loc[[option in item if item==item else False for item in ads_df['wg_type'] ], option_name] = 1
#     ads_df = ads_df.drop(columns=['wg_type'])


    ## tv
    # 1 = yes
    # 0 = no answer (assumed to not exist)
    # NaN = no answer or not searched for details (see details_searched)
    unique_tv = ['Kabel', 'Satellit']

    for option in unique_tv:
        option_name = 'tv_' + option.lower()
        ads_df[option_name] = np.nan
        ads_df.loc[ads_df['details_searched'] == 1.0, option_name] = 0.0
        ads_df.loc[[option in item if item==item else False for item in ads_df['tv'] ], option_name] = 1
#     ads_df = ads_df.drop(columns=['tv'])

    return ads_df

def get_availablility_time(published_on, available_to, available_from):
    '''
    Return the time in days for which an offer will be available
    '''
    if pd.isnull(available_from):
        available_from = published_on

    if pd.isnull(available_to):
        return 365

    return int((available_to-available_from).days)

def feature_engineering(ads_df):
    # Create day of the week column with first 3 letters of the day name
    ads_df['day_of_week_publication'] = ads_df['published_on'].dt.day_name()
    ads_df['day_of_week_publication'] = [day[0:3] for day in list(ads_df['day_of_week_publication'])]

    # Create price/sqm column
    ads_df['price_per_sqm'] = round(ads_df['price_euros']/ads_df['size_sqm'],2)

    # Create available time measured in days
#     ads_df['time_available'] = ads_df.apply(lambda x: print(x['published_on']), axis = 1)

    ads_df['days_available'] = ads_df.apply(lambda x: \
        get_availablility_time(published_on=x['published_on'],
                               available_to=x['available_to'],
                               available_from=x['available_from']), axis = 1)

    return ads_df



if __name__ == "__main__":
    all_ads = get_file(file_name='all_encoded.csv', local_file_path=f'housing_crawler/data').rename(columns={'WG_size':'capacity', 'available from':'available_from', 'available to':'available_to', 'Schufa_needed':'schufa_needed', 'TV':'tv'})


    df_processed = all_ads.copy()
    df_processed = prepare_data(ads_df = df_processed)
    df_processed = filter_out_bad_entries(ads_df = df_processed, country = 'Germany',
                           price_max = 6000, price_min = 50,
                          size_max = 400, size_min = 3,
                          date_max = None, date_min = None, date_format = "%d.%m.%Y")
    # df_processed = transform_columns_into_numerical(ads_df = df_processed)
    # df_processed = feature_engineering(ads_df = df_processed)

    print(df_processed.info())


    # sorted(df_processed.columns, reverse=False)
