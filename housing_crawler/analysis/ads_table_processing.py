import re
import time
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'. Needed to remove SettingWithCopyWarning warning when assigning new value to dataframe column

from shapely.geometry import Point, Polygon
import geopandas as gpd
from shapely import wkt


from housing_crawler.utils import save_file, get_file, get_grid_polygons_all_cities


def prepare_data(ads_df):
    '''
    This function fixes minor mistakes in the table and correctly sets the data types
    '''
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

    return ads_df

def filter_out_bad_entries(ads_df, country = 'Germany',
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
    ads_df['keep'] = False
    ads_df['keep'] = ads_df.apply(lambda x: True if (x['type_offer_simple'] == "WG"\
                    and x['price_euros'] <= 3000\
                    and x['price_euros'] > 100\
                    and x['size_sqm'] <= 60\
                    and x['size_sqm'] >= 5) else x['keep'], axis=1)

    ads_df['keep'] = ads_df.apply(lambda x: True if (x['type_offer_simple'] == "Single-room flat"\
                    and x['price_euros'] <= 3000\
                    and x['price_euros'] > 100\
                    and x['size_sqm'] <= 100\
                    and x['size_sqm'] >= 10) else x['keep'], axis=1)

    ads_df['keep'] = ads_df.apply(lambda x: True if (x['type_offer_simple'] == "Apartment"\
                    and x['price_euros'] <= 6000\
                    and x['price_euros'] > 200\
                    and x['size_sqm'] <= 300\
                    and x['size_sqm'] >= 25) else x['keep'], axis=1)
    ads_df = ads_df[ads_df['keep']].drop(columns=['keep'])

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


    ## schufa_needed
    # Schufa requested for rental
    # 1 = requested
    # 0 = not requested (no response)
    # NaN = not searched for details (see details_searched)
    ads_df['schufa_needed'] = [0 if item != item else 1 for item in ads_df['schufa_needed']]
    ads_df.loc[ads_df['details_searched'] == 0, 'schufa_needed'] = np.nan

    ## commercial_landlord
    # 1 = commercial
    # 0 = private
    # NaN = no answer
    mapping_dict = {'Private':0, 'Verifiziert':1}
    ads_df['commercial_landlord']= ads_df['commercial_landlord'].map(mapping_dict)


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

    ## Number of languages
    # 1 if no answer was given
    # NaN = not searched for details (see details_searched)
    ads_df['number_languages'] = ads_df['languages'].apply(lambda x: len(str(x).split(',')) if x == x else 1)


    ## smoking
    # 1 = allowed everywhere
    # 0.75 = allowed in room
    # 0.5 = allowed in the balcony (outside)
    # 0 = not allowed or no response
    # NaN = not searched for details (see details_searched)
    mapping_dict = {'Rauchen nicht erwünscht':0, 'Rauchen auf dem Balkon erlaubt':0.5, 'Rauchen im Zimmer erlaubt':0.75, 'Rauchen überall erlaubt':1}
    ads_df['smoking']= ads_df['smoking'].map(mapping_dict).replace(np.nan,0)
    ads_df.loc[ads_df['details_searched'] == 0, 'smoking'] = np.nan



    return ads_df

def my_column_splitter(df, cat_name, unique_terms, drop = False):

    for term in unique_terms:
        term_name = cat_name + '_' + term.lower().replace('ü','ue').replace('-wg','').replace(' wg','').replace('wg ','')\
        .replace('ä','ae').replace(' ','_').replace('/','_').replace('-','_').replace('+','')
        df[term_name] = np.nan
        df.loc[df['details_searched'] == 1.0, term_name] = 0
        df.loc[[term in item if item==item else False for item in df[cat_name] ], term_name] = 1

    if drop:
        df = df.drop(columns=[cat_name])
    return df

def split_cat_columns(ads_df):
    '''
    Convert columsn with several terms per row into individual rows.
    '''

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

def feature_engineering(ads_df):
    # Create day of the week column with first 3 letters of the day name
    ads_df['day_of_week_publication'] = ads_df['published_on'].dt.day_name()
    ads_df['day_of_week_publication'] = [day[0:3] for day in list(ads_df['day_of_week_publication'])]

    ## Create price/sqm column
    # for single and multi room flats and houses, the €/m² is directly calculated.
    ads_df['price_per_sqm'] = round(ads_df['price_euros']/ads_df['size_sqm'],2)
    # For WGs, I assume that all rooms in the flat have the same size, so I obtain total size of the flat by multiplying the room size by the number of rooms (plus one (corresponding to the kitchen)).
    # Effectivelly the assumption about the flat size seems valid as the mean size of the offered room over a large number of ads tends to the mean of all rooms in flats. That is unless there are biases with people generally offering the smallest room in the flat, which could very well be the case.
    # ads_df["price_per_sqm"] = ads_df.apply(lambda x: x["price_per_sqm"]/(x["capacity"]) if x["type_offer_simple"] == 'WG' else x["price_per_sqm"], axis = 1)

    # Create available time measured in days
    ads_df['days_available'] = ads_df.apply(lambda x: \
        get_availablility_time(published_on=x['published_on'],
                               available_to=x['available_to'],
                               available_from=x['available_from']), axis = 1)

    # Create availability term
    # very long term (>=540 days)
    # long term (>=365 and <540 days)
    # mid-long term (>=270 and <365 days)
    # mid term (>=180 and <270 days)
    # mid-short term (>=90 and <180 days)
    # short term (<90 days)
    ads_df['rental_length_term'] = ads_df['days_available'].apply(lambda x: '<90days' if x<90 else '<180days' if x<180 else '<270days' if x<270 else '<365days' if x<365 else '<540days' if x<540 else '>=540days')



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




    save_file(ads_df, file_name='ads_OSM.csv', local_file_path=f'housing_crawler/data')
    return ads_df

def imputing_values(ads_df):

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
    ads_df['languages_deutsch'] = ads_df.apply(lambda x: 1 if x['number_languages']!= x['number_languages'] else x['languages_deutsch'], axis =1)
    ads_df.loc[ads_df['details_searched'] == 0, 'languages_deutsch'] = np.nan

    ads_df['number_languages'] = ads_df.number_languages.replace(np.nan, 0)
    ads_df.loc[ads_df['details_searched'] == 0, 'number_languages'] = np.nan

    return ads_df

def get_processed_ads_table(update_table=False):
    # try:
    #     if ~update_table:
    #         return get_file(file_name='ads_OSM.csv', local_file_path=f'housing_crawler/data')
    # except FileNotFoundError:
        all_ads = get_file(file_name='all_encoded.csv', local_file_path='housing_crawler/data')

        df_processed = prepare_data(ads_df = all_ads)
        df_processed = filter_out_bad_entries(ads_df = df_processed, country = 'Germany',
                            date_max = None, date_min = None, date_format = "%d.%m.%Y")

        df_processed = transform_columns_into_numerical(ads_df = df_processed)
        df_processed = split_cat_columns(ads_df = df_processed)
        df_processed = feature_engineering(ads_df = df_processed)
        df_processed = imputing_values(ads_df = df_processed)

        df_processed = df_processed.drop_duplicates(subset = ['id'], keep='first')

        return df_processed


if __name__ == "__main__":

    df_processed = get_processed_ads_table()

    # print(df_processed.info())


    # sorted(df_processed.columns, reverse=False)
