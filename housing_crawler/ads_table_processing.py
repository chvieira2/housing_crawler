import re
import time
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'. Needed to remove SettingWithCopyWarning warning when assigning new value to dataframe column
import statsmodels.api as sm

from shapely.geometry import Point
import geopandas as gpd


from housing_crawler.utils import save_file, get_file, get_grid_polygons_all_cities
from housing_crawler.string_utils import standardize_characters


def prepare_data(ads_df):
    '''
    This function fixes minor mistakes in the table and correctly sets the data types
    '''

    print('Preparing data...')
    ## Minor mistakes
    ads_df = ads_df.rename(columns={'WG_size':'capacity',
                                            'available from':'available_from',
                                            'available to':'available_to',
                                            'Schufa_needed':'schufa_needed',
                                             'TV':'tv',
                                             'landlord_type':'commercial_landlord'})

    ads_df['commercial_landlord'] = ads_df['commercial_landlord'].replace('s','Verifiziert').replace('VerifiziertesUnternehmen','Verifiziert')


    ## Preapare data types
    ads_df['published_at'] = ads_df['published_at'].astype('Int64') # Int64 can take NaN while int or int64 won't
    ads_df['published_on'] = pd.to_datetime(ads_df['published_on'], infer_datetime_format=True).dt.strftime('%d.%m.%Y') # I need to correct format for some unknown reason (same column with multiple format 01.12.2022 and 2022-12-01)
    ads_df['published_on'] = pd.to_datetime(ads_df['published_on'], format = "%d.%m.%Y")
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



    ## Cold rent
    # Filter type of offer
    wg_df = ads_df.query('type_offer_simple == "WG"').reset_index().drop(columns=['index'])

    singleroom_df = ads_df.query('type_offer_simple == "Single-room flat"').reset_index().drop(columns=['index'])

    flathouse_df = ads_df.query('(type_offer_simple == "Apartment")').reset_index().drop(columns=['index'])

    # Exclude cold rent values equal or above warm rent
    if len(wg_df) >0:
        wg_df['price_euros'] = wg_df.apply(lambda x: max(x['price_euros'],x['price_euros']), axis=1)
        wg_df['cold_rent_euros'] = wg_df.apply(lambda x: x['cold_rent_euros'] if x['cold_rent_euros'] < x['price_euros'] else np.nan, axis=1)

    if len(singleroom_df) >0:
        singleroom_df['price_euros'] = singleroom_df.apply(lambda x: max(x['price_euros'],x['price_euros']), axis=1)
        singleroom_df['cold_rent_euros'] = singleroom_df.apply(lambda x: x['cold_rent_euros'] if x['cold_rent_euros'] < x['price_euros'] else np.nan, axis=1)

    if len(flathouse_df) >0:
        flathouse_df['price_euros'] = flathouse_df.apply(lambda x: max(x['price_euros'],x['price_euros']), axis=1)
        flathouse_df['cold_rent_euros'] = flathouse_df.apply(lambda x: x['cold_rent_euros'] if x['cold_rent_euros'] < x['price_euros'] else np.nan, axis=1)

    # create predictive model for cold rent from warm rent
    # wg_df_foo = wg_df[wg_df['cold_rent_euros'].notna()]
    # wg_df_foo = wg_df_foo[wg_df_foo['price_euros'].notna()]
    # model_wg = sm.OLS(wg_df_foo.cold_rent_euros, wg_df_foo.price_euros).fit()

    # singleroom_df_foo = singleroom_df[singleroom_df['cold_rent_euros'].notna()]
    # singleroom_df_foo = singleroom_df_foo[singleroom_df_foo['price_euros'].notna()]
    # model_single = sm.OLS(singleroom_df_foo.cold_rent_euros, singleroom_df_foo.price_euros).fit()

    # flathouse_df_foo = flathouse_df[flathouse_df['cold_rent_euros'].notna()]
    # flathouse_df_foo = flathouse_df_foo[flathouse_df_foo['price_euros'].notna()]
    # model_multi = sm.OLS(flathouse_df_foo.cold_rent_euros, flathouse_df_foo.price_euros).fit()

    # # Add cold rent predictions
    # wg_df['cold_rent_euros'] = wg_df.apply(lambda x: x['cold_rent_euros'] if x['cold_rent_euros'] == x['cold_rent_euros'] else np.nan if x['price_euros']!=x['price_euros'] else round(model_wg.predict(x['price_euros'])[0],0), axis=1)

    # singleroom_df['cold_rent_euros'] = singleroom_df.apply(lambda x: x['cold_rent_euros'] if x['cold_rent_euros'] == x['cold_rent_euros'] else np.nan if x['price_euros']!=x['price_euros'] else round(model_single.predict(x['price_euros'])[0],0), axis=1)

    # flathouse_df['cold_rent_euros'] = flathouse_df.apply(lambda x: x['cold_rent_euros'] if x['cold_rent_euros'] == x['cold_rent_euros'] else np.nan if x['price_euros']!=x['price_euros'] else round(model_multi.predict(x['price_euros'])[0],0), axis=1)

    # Concatenate all ads together to re-form ads_df
    ads_df = pd.concat([wg_df, singleroom_df, flathouse_df], axis=0)





    ## Age range flatmates
    # min age range flatmates
    ads_df['min_age_flatmates'] = np.nan
    ads_df['min_age_flatmates'] = [np.nan if item != item else \
                                   np.nan if str(item).startswith('bis') else \
                               float(re.findall('[0-9]+', item)[0]) \
                               for item in list(ads_df['age_range'])]
    # max age range flatmates
    ads_df['max_age_flatmates'] = np.nan
    ads_df['max_age_flatmates'] = [np.nan if item != item else \
                                   np.nan if str(item).startswith('ab') else \
                                   float(re.findall('[0-9]+', item)[0]) if str(item).startswith('bis') else \
                               float(re.findall('[0-9]+', item)[1]) \
                               for item in list(ads_df['age_range'])]
    ads_df = ads_df.drop(columns=['age_range'])



    ## gender_search
    # gender searched
    ads_df['gender_searched'] = np.nan
    ads_df['gender_searched'] = ['Egal' if item != item else \
                                'Divers' if 'Divers' in item else \
                                'Frau' if 'Frau' in item else \
                                'Mann' if 'Mann' in item else \
                                'Egal' \
                                for item in list(ads_df['gender_search'])]
    ads_df.loc[ads_df['details_searched'] == 0, 'gender_searched'] = np.nan




    ## Age searched
    # min age range searched
    ads_df['min_age_searched'] = np.nan
    ads_df['min_age_searched'] = [0 if item != item else \
                                   0 if 'bis' in item else \
                                   int(min(re.findall('[0-9]+', item))) if 'zwischen' in item else \
                               int(re.findall('[0-9]+', item)[0]) if 'ab' in item else 0\
                               for item in list(ads_df['gender_search'])]
    ads_df.loc[ads_df['details_searched'] == 0, 'min_age_searched'] = np.nan




    # max age range searched
    ads_df['max_age_searched'] = np.nan
    ads_df['max_age_searched'] = [99 if item != item else \
                                   99 if 'ab' in item else \
                                   int(max(re.findall('[0-9]+', item))) if 'zwischen' in item else \
                               int(re.findall('[0-9]+', item)[0]) if 'bis' in item else 99\
                               for item in list(ads_df['gender_search'])]
    ads_df.loc[ads_df['details_searched'] == 0, 'max_age_searched'] = np.nan

    ads_df = ads_df.drop(columns=['gender_search'])




    ## energy
    ads_df['construction_year'] = np.nan
    ads_df['construction_year'] = [np.nan if item != item else \
        int(str(item).split('Baujahr ')[1].split(',')[0]) if 'Baujahr ' in item else np.nan \
        for item in list(ads_df['energy'])]
    ads_df.loc[ads_df['details_searched'] == 0, 'construction_year'] = np.nan


    ads_df['energy_certificate'] = np.nan
    ads_df['energy_certificate'] = [np.nan if item != item else \
        'Verbrauchsausweis' if 'Verbrauchsausweis' in item else \
        'Bedarfsausweis' if 'Bedarfsausweis' in item else np.nan \
        for item in list(ads_df['energy'])]
    ads_df.loc[ads_df['details_searched'] == 0, 'energy_certificate'] = np.nan


    ads_df['energy_usage'] = np.nan
    ads_df['energy_usage'] = [np.nan if item != item else \
        int(str(item).split('V: ')[1].split('kW h/')[0]) if 'kW h/' in item else np.nan \
        for item in list(ads_df['energy'])]
    ads_df.loc[ads_df['details_searched'] == 0, 'energy_usage'] = np.nan


    ads_df['energy_efficiency_class'] = np.nan
    ads_df['energy_efficiency_class'] = [np.nan if item != item else \
        str(item).split('Energieeffizienzklasse ')[1].split(',')[0] if 'Energieeffizienzklasse' in item else np.nan \
        for item in list(ads_df['energy'])]
    ads_df.loc[ads_df['details_searched'] == 0, 'energy_efficiency_class'] = np.nan


    ads_df['heating_energy_source'] = np.nan
    ads_df['heating_energy_source'] = [np.nan if item != item else \
        'oil' if 'Öl' in item else \
        'geothermal' if 'Erdwärme' in item else \
        'solar' if 'Solar' in item else \
        'wood pellets' if 'Holzpellets' in item else \
        'gas' if 'Gas' in item else \
        'steam district heating' if 'Fernwärme-Dampft' in item else \
        'distant district heating' if 'Fernwärme' in item else \
        'coal/coke' if 'Kohle/Koks' in item else \
        'coal' if 'Kohle' in item else \
        'light natural gas' if 'Erdgas leicht' in item else \
        'heavy natural gas' if 'Erdgas schwer' in item else \
        'LPG' if 'Flüssiggas' in item else \
        'wood' if 'Holz' in item else \
        'wood chips' if 'Holz-Hackschnitzel' in item else \
        'local district heating' if 'Nahwärme' in item else \
        'delivery' if 'Wärmelieferung' in item else \
        'eletricity' if 'Strom' in item else np.nan \
        for item in list(ads_df['energy'])]
    ads_df.loc[ads_df['details_searched'] == 0, 'heating_energy_source'] = np.nan

    ads_df = ads_df.drop(columns=['energy'])

    return ads_df

def filter_out_bad_entries(ads_df, country = 'Germany',
                          date_format = "%d.%m.%Y",
                          filter_time = False):

    print('Filtering out bad entries...')

    if filter_time: # This temporal selection has to be removed for working with files per month
        ## Select offers inside a time period
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

        # Searches before August are not so useful. Remove them here
        # ads_df = ads_df[ads_df['published_on'] >= '2022-08-01']





    ###### Filter out unrealistic offers based on cold_rent_price/size_sqm
    ## Exclude first based on direct rules
    ads_df['keep'] = False
    ads_df['keep'] = ads_df.apply(lambda x: True if (x['type_offer_simple'] == "WG"\
                    and x['price_euros'] <= 2000\
                    and x['price_euros'] >= 50\
                    and x['size_sqm'] <= 60\
                    and x['size_sqm'] >= 5) else x['keep'], axis=1)

    ads_df['keep'] = ads_df.apply(lambda x: True if (x['type_offer_simple'] == "Single-room flat"\
                    and x['price_euros'] <= 2500\
                    and x['price_euros'] >= 100\
                    and x['size_sqm'] <= 100\
                    and x['size_sqm'] >= 10) else x['keep'], axis=1)

    ads_df['keep'] = ads_df.apply(lambda x: True if (x['type_offer_simple'] == "Apartment"\
                    and x['price_euros'] <= 6000\
                    and x['price_euros'] >= 200\
                    and x['size_sqm'] <= 300\
                    and x['size_sqm'] >= 25) else x['keep'], axis=1)
    # Exclude offers not selected in 'keep' column
    ads_df = ads_df[ads_df['keep']].drop(columns=['keep'])

    ## Create price/sqm column
    # for single and multi room flats and houses, the €/m² is directly calculated.
    ads_df['price_per_sqm_warm'] = round(ads_df['price_euros']/ads_df['size_sqm'],2)
    ads_df['price_per_sqm_cold'] = round(ads_df['cold_rent_euros']/ads_df['size_sqm'],2)
    # For WGs, I assume that all rooms in the flat have the same size, so I obtain total size of the flat by multiplying the room size by the number of rooms (plus one corresponding to the kitchen and bathroom).
    # Effectivelly the assumption about the flat size seems valid as the mean size of the offered room over a large number of ads tends to the mean of all rooms in flats. That is unless there are biases with people generally offering the smallest room in the flat, which could very well be the case.

    try:
        ads_df["price_per_sqm_warm"] = ads_df.apply(lambda x: x["price_euros"]*x["capacity"]/x["home_total_size"] if x["type_offer_simple"] == 'WG' else x["price_per_sqm_cold"], axis = 1)
    except:
        ads_df["price_per_sqm_warm"] = np.nan

    try:
        ads_df["price_per_sqm_cold"] = ads_df.apply(lambda x: x["cold_rent_euros"]*x["capacity"]/x["home_total_size"] if x["type_offer_simple"] == 'WG' else x["price_per_sqm_cold"], axis = 1)
    except:
        ads_df["price_per_sqm_cold"] = np.nan



    # ## Split type of offer
    # wg_df = ads_df.query('type_offer_simple == "WG"').reset_index().drop(columns=['index'])

    # singleroom_df = ads_df.query('type_offer_simple == "Single-room flat"').reset_index().drop(columns=['index'])

    # flathouse_df = ads_df.query('(type_offer_simple == "Apartment")').reset_index().drop(columns=['index'])


    # ## Transform prices into log to make it normally distributed
    # wg_df = wg_df[wg_df['price_per_sqm_cold'].notna()]
    # wg_df = wg_df[wg_df['price_per_sqm_cold']>0]
    # wg_df = wg_df[wg_df['price_per_sqm_cold']<np.inf]
    # wg_df['log_price_per_sqm_cold'] = np.log2(wg_df['price_per_sqm_cold'])

    # singleroom_df = singleroom_df[singleroom_df['price_per_sqm_cold'].notna()]
    # singleroom_df = singleroom_df[singleroom_df['price_per_sqm_cold']>0]
    # singleroom_df = singleroom_df[singleroom_df['price_per_sqm_cold']<np.inf]
    # singleroom_df['log_price_per_sqm_cold'] = np.log2(singleroom_df['price_per_sqm_cold'])

    # flathouse_df = flathouse_df[flathouse_df['price_per_sqm_cold'].notna()]
    # flathouse_df = flathouse_df[flathouse_df['price_per_sqm_cold']>0]
    # flathouse_df = flathouse_df[flathouse_df['price_per_sqm_cold']<np.inf]
    # flathouse_df['log_price_per_sqm_cold'] = np.log2(flathouse_df['price_per_sqm_cold'])


    # # ## Remove outliers identified above X standard deviations
    # # X=3
    # # mean_wg_df = np.mean(wg_df['log_price_per_sqm_cold'])
    # # std_wg_df = np.std(wg_df['log_price_per_sqm_cold'],ddof=1)
    # # wg_df = wg_df[wg_df['log_price_per_sqm_cold']<=mean_wg_df+(X*std_wg_df)]
    # # wg_df = wg_df[wg_df['log_price_per_sqm_cold']>=mean_wg_df-(X*std_wg_df)]

    # # mean_singleroom_df = np.mean(singleroom_df['log_price_per_sqm_cold'])
    # # std_singleroom_df = np.std(singleroom_df['log_price_per_sqm_cold'],ddof=1)
    # # singleroom_df = singleroom_df[singleroom_df['log_price_per_sqm_cold']<=mean_singleroom_df+(X*std_singleroom_df)]
    # # singleroom_df = singleroom_df[singleroom_df['log_price_per_sqm_cold']>=mean_singleroom_df-(X*std_singleroom_df)]

    # # mean_flathouse_df = np.mean(flathouse_df['log_price_per_sqm_cold'])
    # # std_flathouse_df = np.std(flathouse_df['log_price_per_sqm_cold'],ddof=1)
    # # flathouse_df = flathouse_df[flathouse_df['log_price_per_sqm_cold']<=mean_flathouse_df+(X*std_flathouse_df)]
    # # flathouse_df = flathouse_df[flathouse_df['log_price_per_sqm_cold']>=mean_flathouse_df-(X*std_flathouse_df)]


    # # Concatenate all ads together to re-form ads_df
    # ads_df = pd.concat([wg_df, singleroom_df, flathouse_df], axis=0)
    # ads_df = ads_df.drop(columns=['log_price_per_sqm_cold'])




    ## Exclude offers with addresses outside of Germany
    if country.lower() in ['germany', 'de']:
        # Germany bounding box coordinates from here: https://gist.github.com/graydon/11198540
        ads_df['latitude'] = [lat if (lat>47.3024876979 and lat<54.983104153) else np.nan for lat in list(ads_df['latitude'])]
        ads_df['longitude'] = [lon if (lon>5.98865807458 and lon<15.0169958839) else np.nan for lon in list(ads_df['longitude'])]

    ads_df.reset_index(inplace=True, drop = True)

    return ads_df

def transform_columns_into_numerical(ads_df):

    print('Transforming columns into numerical...')

    ## wg_possible
    # Only relevant for houses and flats
    # 1 = allowed to turn into WG
    # 0 = not allowed to turn into WG (no response)
    # NaN = not searched for details (see details_searched)
    ads_df['wg_possible'] = [0.0 if item != item else 1.0 for item in ads_df['wg_possible']] # np.nan doesn't equals itself
    ads_df.loc[ads_df['details_searched'] == 0, 'wg_possible'] = np.nan
    ads_df.loc[ads_df['type_offer_simple'] == 'WG', 'wg_possible'] = 1.0


    ## schufa_needed
    # Schufa requested for rental
    # 1 = requested
    # 0 = not requested (no response)
    # NaN = not searched for details (see details_searched)
    ads_df['schufa_needed'] = [0.0 if item != item else 1.0 for item in ads_df['schufa_needed']]
    ads_df.loc[ads_df['details_searched'] == 0, 'schufa_needed'] = np.nan

    ## commercial_landlord
    # 1 = commercial
    # 0 = private
    # NaN = no answer
    mapping_dict = {'Private':0.0, 'Verifiziert':1.0}
    ads_df['commercial_landlord']= ads_df['commercial_landlord'].map(mapping_dict)


    ## energy_efficiency_class
    # 9 = A+
    # 8 = A
    # 7 = B
    # 6 = C
    # 5 = D
    # 4 = E
    # 3 = F
    # 2 = G
    # 1 = H
    # NaN = no answer
    mapping_dict = {'H':1, 'G':2, 'F':3, 'E':4, 'D':5, 'C':6, 'B':7, 'A':8, 'A+':9}
    ads_df['energy_efficiency_class']= ads_df['energy_efficiency_class'].map(mapping_dict)

    ## building_floor
    # indicates the level from the ground. Ground level is 0.
    # Ambiguous values were given fractional definitions ('Hochparterre':0.5, 'Tiefparterre':-0.5).
    # 6 indicates values above 5, not necessarily the 6th floor
    # NaN = indicates lack of response or not searched for details (see details_searched)
    mapping_dict = {'EG':0, '1. OG':1, '2. OG':2, '3. OG':3, '4. OG':4, '5. OG':5, 'höher als 5. OG':6,
                    'Hochparterre':0.5, 'Dachgeschoss':2, 'Tiefparterre':-0.5, 'Keller':-1}
    ads_df['building_floor']= ads_df['building_floor'].map(mapping_dict)


    ## public_transport_distance
    # Distance in minutes to public transportation
    # NaN = indicates lack of response or not searched for details (see details_searched)
    ads_df['public_transport_distance'] = ads_df['public_transport_distance'].apply(lambda x: np.nan if x!=x else int(x.split(' Min')[0]))

    ## Number of languages
    # 1 if no answer was given
    # NaN = not searched for details (see details_searched)
    ads_df['number_languages'] = ads_df['languages'].apply(lambda x: len(str(x).split(',')) if x == x else 1.0)


    ## internet_speed
    # 1 = '<10 Mbit/s' or '1-3 Mbit/s'
    # 2 = '7-10 Mbit/s'
    # 3 = '11-16 Mbit/s'
    # 4 = '17-25 Mbit/s'
    # 5 = '26-50 Mbit/s'
    # 6 = '50-100 Mbit/s'
    # 7 = '>100 Mbit/s'
    # NaN = no response or not searched for details (see details_searched)
    ads_df['internet_speed'] = np.nan
    ads_df['internet_speed'] = [np.nan if item != item else \
        1 if 'langsamer als 10 Mbit/s' in item else \
        1 if '1-3 Mbit/s' in item else \
        2 if '7-10 Mbit/s' in item else \
        3 if '11-16 Mbit/s' in item else \
        4 if '17-25 Mbit/s' in item else \
        5 if '26-50 Mbit/s' in item else \
        6 if '50-100 Mbit/s' in item else \
        7 if 'schneller als 100 Mbit/s' in item else np.nan \
        for item in list(ads_df['internet'])]
    ads_df.loc[ads_df['details_searched'] == 0, 'internet_speed'] = np.nan


    ## toilet
    # 1 = individual
    # 0.5 = shared
    # 0 = not available
    # NaN = no response or not searched for details (see details_searched)
    ads_df['toilet'] = np.nan
    ads_df['toilet'] = [np.nan if item != item else \
        1.0 if 'Eigenes Bad' in item else \
        0.5 if 'Badmitbenutzung' in item else \
        0.0 if 'Nicht vorhanden' in item else np.nan \
        for item in list(ads_df['shower_type'])]
    ads_df.loc[ads_df['details_searched'] == 0, 'toilet'] = np.nan



    return ads_df

def my_column_splitter(df, cat_name, unique_terms, drop = False):

    for term in unique_terms:
        term_name = cat_name + '_' + term.lower().replace('ü','ue').replace('-wg','').replace(' wg','').replace('wg ','')\
        .replace('ä','ae').replace(' ','_').replace('/','_').replace('-','_').replace('+','')
        df[term_name] = np.nan
        df.loc[df['details_searched'] == 1.0, term_name] = 0.0
        df.loc[[term in item if item==item else False for item in df[cat_name] ], term_name] = 1.0

    if drop:
        df = df.drop(columns=[cat_name])
    return df

def split_cat_columns(ads_df):
    '''
    Convert columsn with several terms per row into individual rows.
    '''

    print('Splitting categorical columns...')

    ## internet
    ads_df = my_column_splitter(df=ads_df,
                           cat_name = 'internet',
                           unique_terms = ['DSL', 'WLAN', 'Flatrate'],
                           drop = True)

    ## shower_type
    ads_df = my_column_splitter(df=ads_df,
                           cat_name = 'shower_type',
                           unique_terms = ['Badewanne', 'Dusche'],
                           drop = True)

    ## floor_type
    ads_df = my_column_splitter(df=ads_df,
                           cat_name = 'floor_type',
                           unique_terms = ['Dielen', 'Parkett', 'Laminat', 'Teppich', 'Fliesen', 'PVC', 'Fußbodenheizung'],
                           drop = True)

    ## extras
    ads_df = my_column_splitter(df=ads_df,
                           cat_name = 'extras',
                           unique_terms = ['Waschmaschine', 'Spülmaschine', 'Terrasse',
                                           'Balkon', 'Garten', 'Gartenmitbenutzung', 'Keller', 'Aufzug', 'Haustiere', 'Fahrradkeller', 'Dachboden'],
                           drop = True)


    ## languages
    ads_df = my_column_splitter(df=ads_df,
                           cat_name = 'languages',
                           unique_terms = ['Deutsch', 'Englisch'],
                           drop = True)


    ## wg_type
    ads_df = my_column_splitter(df=ads_df,
                           cat_name = 'wg_type',
                           unique_terms = ['Studenten-WG', 'keine Zweck-WG', 'Männer-WG', 'Business-WG', 'Wohnheim', 'Vegetarisch/Vegan', 'Alleinerziehende', 'funktionale WG', 'Berufstätigen-WG', 'gemischte WG', 'WG mit Kindern', 'Verbindung', 'LGBTQIA+', 'Senioren-WG', 'inklusive WG', 'WG-Neugründung'],
                           drop = True)


    ## tv
    ads_df = my_column_splitter(df=ads_df,
                           cat_name = 'tv',
                           unique_terms = ['Kabel', 'Satellit'],
                           drop = True)

    return ads_df

def get_availablility_time(published_on, available_to, available_from):
    '''
    Return the time in days for which an offer will be available
    '''
    if pd.isnull(available_from):
        available_from = published_on

    if pd.isnull(available_to):
        return 365*2

    return int((available_to-available_from).days)

def feature_engineering(ads_df, df_feats = None):

    print('Engineering features...')

    # Create day of the week column with first 3 letters of the day name
    ads_df['day_of_week_publication'] = ads_df['published_on'].dt.day_name()
    ads_df['day_of_week_publication'] = [day[0:3] for day in list(ads_df['day_of_week_publication'])]


    # Create available time measured in days
    try:
        ads_df['days_available'] = ads_df.apply(lambda x: \
            get_availablility_time(published_on=x['published_on'],
                               available_to=x['available_to'],
                               available_from=x['available_from']), axis = 1)
    except:
        ads_df['days_available'] = np.nan

    # Create availability term
    # very long term (>=540 days)
    # long term (>=365 and <540 days)
    # mid-long term (>=270 and <365 days)
    # mid term (>=180 and <270 days)
    # mid-short term (>=90 and <180 days)
    # short term (<90 days)
    # very short term (<30 days)
    ads_df['rental_length_term'] = ads_df['days_available'].apply(lambda x: '30days' if x<=30 else '90days' if x<=90 else '180days' if x<=180 else '270days' if x<=270 else '365days' if x<365 else '540days' if x<540 else 'plus540days')


    ## furniture
    # 1 = möbliert
    # 0.5 = teilmöbliert
    # 0 = no answer (assumed to be not furnitured)
    # NaN = not searched for details (see details_searched)
    mapping_dict = {'möbliert':1.0, 'teilmöbliert':0.5, 'möbliert, teilmöbliert':0.5}
    ads_df['furniture_numerical']= ads_df['furniture'].map(mapping_dict).replace(np.nan,0.0)
    ads_df.loc[ads_df['details_searched'] == 0, 'furniture_numerical'] = np.nan


    ## kitchen
    # 1 = kitchen ('Eigene Küche' or 'Einbauküche')
    # 0.75 = 'Kochnische' (room + kitchen)
    # 0.5 = 'Küchenmitbenutzung' (shared kitchen)
    # 0 = no kitchen (Nicht vorhanden [not available] or no response)
    # NaN = not searched for details (see details_searched)
    mapping_dict = {'Nicht vorhanden':0.0, 'Küchenmitbenutzung':0.5, 'Kochnische':0.75, 'Eigene Küche':1.0, 'Einbauküche':1.0}
    ads_df['kitchen_numerical']= ads_df['kitchen'].map(mapping_dict).replace(np.nan,0.0)
    ads_df.loc[ads_df['details_searched'] == 0, 'kitchen_numerical'] = np.nan


    ## smoking_numerical
    # 1 = allowed everywhere
    # 0.75 = allowed in room
    # 0.5 = allowed in the balcony (outside)
    # 0 = not allowed or no response
    # NaN = not searched for details (see details_searched)
    mapping_dict = {'Rauchen nicht erwünscht':0.0, 'Rauchen auf dem Balkon erlaubt':0.5, 'Rauchen im Zimmer erlaubt':0.75, 'Rauchen überall erlaubt':1.0}
    ads_df['smoking_numerical']= ads_df['smoking'].map(mapping_dict).replace(np.nan,0.0)
    ads_df.loc[ads_df['details_searched'] == 0, 'smoking_numerical'] = np.nan



    #### Searched flatmate age
    # senior (>=60 years old)
    # mature (>=40 and <60 years old)
    # adult (>=30 and <40 years old)
    # young (<30 years old)
    # teen (<20 years old)
    ads_df['max_age_searched_encoded'] = ads_df['max_age_searched'].apply(lambda x: '20' if x<20 else '30' if x<30 else '40' if x<40 else '60' if x<60 else '100')
    ads_df['min_age_searched_encoded'] = ads_df['min_age_searched'].apply(lambda x: '20' if x<20 else '30' if x<30 else '40' if x<40 else '60' if x<60 else '100')
    ads_df['age_category_searched'] = ads_df['min_age_searched_encoded'] + '_' + ads_df['max_age_searched_encoded']

    ads_df = ads_df.drop(columns=['min_age_searched_encoded', 'max_age_searched_encoded'])


    ## room_size_house_fraction
    ads_df['room_size_house_fraction'] = ads_df['size_sqm']/ads_df['home_total_size']
    # Room can not occupy more than 70% of the flat/house area
    ads_df = ads_df.loc[(ads_df['room_size_house_fraction']<=0.7) | ~ads_df['room_size_house_fraction'].notnull()]




    #### Manage identified outliers
    # max_age_flatmates
    # There are some ads with really extreme energy usage values (<18 or >80). These were removed for modelling
    ads_df['max_age_flatmates'] = [np.nan if (value!=value) or value < 18 or value >80 else value for value in list(ads_df['max_age_flatmates'])]

    # public_transport_distance
    # Remove values of public transport above 30 min
    # ads_df['public_transport_distance'] = [np.nan if (value!=value) or value > 30 else value for value in list(ads_df['public_transport_distance'])]

    # min_age_flatmates
    # Create a new variable flat_with_kids if the minimum age of flatmates is below 18
    ads_df['flat_with_kids'] = [1.0 if value < 18 else 0.0 for value in list(ads_df['min_age_flatmates'])]
    # Limit minimal age of flat mates to 18 and 80 yo
    ads_df['min_age_flatmates'] = [np.nan if (value!=value) or value < 18 or value >80 else value for value in list(ads_df['min_age_flatmates'])]




    ########### Collect OpenStreetMaps features
    # Filter for only ads with identifiable coordinates
    ads_df = ads_df[ads_df['latitude'].notna()].reset_index(drop=True)
    ads_df = ads_df[ads_df['longitude'].notna()].reset_index(drop=True)

    ads_df = ads_df[ads_df['latitude'] > 0]
    ads_df = ads_df[ads_df['longitude'] > 0]

    ads_df['coords'] = list(zip(ads_df['latitude'],ads_df['longitude']))
    ads_df['geometry'] = ads_df['coords'].apply(Point)
    ads_df = gpd.GeoDataFrame(ads_df, geometry='geometry', crs='wgs84').drop(columns=['coords'])

    # Collect features
    if df_feats is None:
        df_feats = get_grid_polygons_all_cities()

    # Merge features into ads dataframe
    ads_df = gpd.sjoin(ads_df,df_feats,how="left").drop(columns=['geometry', 'index_right'])





    #### Polar transformation #####
    # degrees_to_centroid
    degrees = 360
    ads_df['sin_degrees_to_centroid'] = np.sin(2*np.pi*ads_df.degrees_to_centroid/degrees)
    ads_df['cos_degrees_to_centroid'] = np.cos(2*np.pi*ads_df.degrees_to_centroid/degrees)

    ads_df.drop('degrees_to_centroid', axis=1, inplace=True)

    # published_at = hour of the day
    hours_day = 24
    ads_df['sin_published_at'] = np.sin(2*np.pi*ads_df.published_at/hours_day)
    ads_df['cos_published_at'] = np.cos(2*np.pi*ads_df.published_at/hours_day)

    # day_of_week_publication = day of the week of publication
    ads_df['day_week_int'] = ads_df['day_of_week_publication'].map({'Mon':1,
                                                                    'Tue':2,
                                                                    'Wed':3,
                                                                    'Thu':4,
                                                                    'Fri':5,
                                                                    'Sat':6,
                                                                    'Sun':7})
    days_week = 7
    ads_df['sin_day_week_int'] = np.sin(2*np.pi*ads_df.day_week_int/days_week)
    ads_df['cos_day_week_int'] = np.cos(2*np.pi*ads_df.day_week_int/days_week)

    ads_df.drop('day_week_int', axis=1, inplace=True)

    return ads_df

def imputing_values(ads_df):

    print('Imputing values...')

    #### Imputting zero-value when there has been no answer to costs
    ## transfer_costs_euros
    # Contract transfer costs (mostly relevant for flat/houses)
    # Ablösevereinbarung
    # 0 = 0, n.a. response or no response
    # NaN = not searched for details (see details_searched)
    ads_df['transfer_costs_euros'] = ads_df['transfer_costs_euros'].replace(np.nan, 0).fillna(0)
    ads_df.loc[ads_df['details_searched'] == 0, 'transfer_costs_euros'] = np.nan


    ## extra_costs_euros
    # Extra costs
    # Sonstige Kosten
    # 0 = 0, n.a. response or no response
    # NaN = not searched for details (see details_searched)
    ads_df['extra_costs_euros'] = ads_df['extra_costs_euros'].replace(np.nan, 0).fillna(0)
    ads_df.loc[ads_df['details_searched'] == 0, 'extra_costs_euros'] = np.nan


    ## mandatory_costs_euros
    # Living costs (water, heat, internet...)
    # Nebenkosten
    # 0 = 0, n.a. response or no response
    # NaN = not searched for details (see details_searched)
    ads_df['mandatory_costs_euros'] = ads_df['mandatory_costs_euros'].replace(np.nan, 0).fillna(0)
    ads_df.loc[ads_df['details_searched'] == 0, 'mandatory_costs_euros'] = np.nan


    ## deposit
    # Deposit paid at the start of contract
    # Kaution
    # 0 = 0, n.a. response or no response
    # NaN = not searched for details (see details_searched)
    ads_df['deposit'] = ads_df['deposit'].replace(np.nan, 0).fillna(0)
    ads_df.loc[ads_df['details_searched'] == 0, 'deposit'] = np.nan


    ## number_languages
    # Number of languages spoken at home
    # Sprach
    # Assumed every house must speak at least one language. If no languages are given, assume German is spoken.
    # NaN = not searched for details (see details_searched)
    ads_df['languages_deutsch'] = ads_df.apply(lambda x: 1.0 if x['number_languages']!= x['number_languages'] else x['languages_deutsch'], axis =1)
    ads_df.loc[ads_df['details_searched'] == 0, 'languages_deutsch'] = np.nan

    ads_df['number_languages'] = ads_df.number_languages.replace(np.nan, 0)
    ads_df.loc[ads_df['details_searched'] == 0, 'number_languages'] = np.nan

    #### Imputting categorical values when there has been no answer
    ## energy_certificate
    ads_df['energy_certificate'] = ads_df['energy_certificate'].replace(np.nan, 'no_answer')
    ads_df.loc[ads_df['details_searched'] == 0, 'energy_certificate'] = np.nan


    ## heating_energy_source
    ads_df['heating_energy_source'] = ads_df['heating_energy_source'].replace(np.nan, 'no_answer')
    ads_df.loc[ads_df['details_searched'] == 0, 'heating_energy_source'] = np.nan


    ## heating
    ads_df['heating'] = ads_df['heating'].replace(np.nan, 'no_answer')
    ads_df.loc[ads_df['details_searched'] == 0, 'heating'] = np.nan


    ## parking
    ads_df['parking'] = ads_df['parking'].replace(np.nan, 'no_answer')
    ads_df.loc[ads_df['details_searched'] == 0, 'parking'] = np.nan


    ## building_type
    ads_df['building_type'] = ads_df['building_type'].replace(np.nan, 'no_answer')
    ads_df.loc[ads_df['details_searched'] == 0, 'building_type'] = np.nan

    return ads_df

def process_ads_tables(input_ads_df = None, save_processed = True, df_feats = None):

    year = time.strftime(f"%Y", time.localtime())
    month = time.strftime(f"%m", time.localtime())
    if input_ads_df is None:
        input_ads_df = get_file(file_name=f'''{str(year)}{str(month)}_ads_encoded.csv''', local_file_path='raw_data')

    df_processed = prepare_data(ads_df = input_ads_df)
    df_processed = filter_out_bad_entries(ads_df = df_processed, country = 'Germany')

    df_processed = transform_columns_into_numerical(ads_df = df_processed)
    df_processed = split_cat_columns(ads_df = df_processed)
    df_processed = feature_engineering(ads_df = df_processed, df_feats=df_feats)
    df_processed = imputing_values(ads_df = df_processed)

    df_processed = df_processed.drop_duplicates(subset = ['id'], keep='first')

    if save_processed:
        save_file(df_processed, file_name=f'''{year}{month}_ads_OSM.csv''', local_file_path=f'raw_data')

    return df_processed

if __name__ == "__main__":

    # process_ads_tables(2022,12)



    data = {'col_1': [3], 'col_2': 'a'}
    print(time.strftime(f"%H", time.localtime()))
