# -*- coding: utf-8 -*-

"""Housing Crawler - search for flats by crawling property portals and save them locally.

This is the dashboard/app implementation of the analysis of ads obtained from wg-gesucht.de"""

__author__ = "Carlos Henrique Vieira e Vieira"
__version__ = "1.0"
__maintainer__ = "chvieira2"
__email__ = "carloshvieira2@gmail.com"
__status__ = "Production"


import streamlit as st
from housing_crawler.params import dict_city_number_wggesucht
from housing_crawler.utils import crawl_ind_ad_page2,get_grid_polygons_all_cities
from housing_crawler.ads_table_processing import process_ads_tables
from housing_crawler.geocoding_addresses import geocoding_address

import pandas as pd
import numpy as np
import time
import pickle
import geopandas as gpd

from config.config import ROOT_DIR


#-----------------------page configuration-------------------------
st.set_page_config(
    page_title="housing_crawler",
    page_icon=':house:', # gives a random emoji
    layout="wide",
    initial_sidebar_state="auto")

#-------------------styling for layouts--------------------------
#.css-18e3th9 change top padding main container
# .css-1oe6wy4 changed top paging sidebar
# iframe changed the size of the map's iframe
st.markdown(
            f'''
            <style>
                .css-18e3th9 {{
                    padding-top: 15px;
                    padding-right: 15px;
                    padding-bottom: 15px;
                    padding-left: 15px;
                }}
                .css-1oe6wy4 {{
                    padding-top: 15px;
                }}
                .css-192cp98{{
                    padding-top: 15px;
                }}

                iframe {{
                width: 100%;
                height: 500px;
                }}
                .css-1inwz65{{
                    font-family:inherit
                }}
                .css-16huue1{{
                    font-size:18px;
                    color: rgb(139, 145, 153);
                    justify-content: center;
                }}
                .st-bt {{
                    background-color: inherit;
                }}
            </style>
            ''', unsafe_allow_html=True)

# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

#----Content starts------
placeholder_map = st.empty()
with placeholder_map.container():
    st.markdown("""
            ### Have you ever wondered if what you pay for your room in a WG (flatshare) is according to the current market?
            #### I've collected thousands of ads from wg-gesucht.de to answer that question for you.
            """, unsafe_allow_html=True)

with st.expander("I want to know the market price for my room in a WG"):
    st.caption("""
        This information is not stored anywhere and won't be used for anything other than predicting the price.
        The more information you give, the better is the prediction.
                """, unsafe_allow_html=True)
    with st.form("entry_form", clear_on_submit=True):
        placeholder_map = st.empty()
        with placeholder_map.container():
            st.subheader("""
                    \n
                    Location
                    """)
            col1, col2, col3, col4 = st.columns(4)
            col1.selectbox(label="City*", options=['<Please select>']+sorted(list(dict_city_number_wggesucht.keys())), index=0, key='city')
            col2.text_input("Street and house number*", value="Example Str. 15", key='address', max_chars = 100)
            col3.text_input("Neighborhood", value="", key='neighborhood', max_chars = 100)
            col4.text_input("Zip code*", value="12345", key='zip_code', max_chars = 20)

        placeholder_map = st.empty()
        with placeholder_map.container():
            st.subheader("""
                    \n
                    Information about the room and building
                    """)
            col1, col2, col3, col4 = st.columns(4)
            col1.number_input(label='Room size (m²)*', min_value=0, max_value=60, value=0, step=1, key='room_size')
            col2.number_input(label='Total flat/house size (m²)', min_value=0, max_value=250, value=0, step=1, key='total_flat_size')
            col3.selectbox(label="Type of building", options=['<Please select>','Types of building here'], index=0, key='building_type')
            col4.selectbox(label="Floor", options=['<Please select>','Basement', 'Low ground floor','Ground floor','High ground floor','1st floor','2nd floor','3rd floor','4th floor','5th floor','6th floor or higher','Attic'], index=0, key='floor')


            col1.selectbox(label='Parking condition', options=['<Please select>','Good parking facilities', 'Bad parking facilities', 'Resident parking', 'Own parking', 'Underground parking'], index=0, key='parking')
            col2.select_slider(label='Walking distance to public transport (in minutes)', options=[str(n) for n in range(1,61)], value='10', key='distance_public_transport')
            col3.selectbox("Barrier-free", ['<Please select>','Suitable for wheelchair','Not suitable for wheelchair'], index=0, key='barrier_free')
            col4.selectbox("Schufa requested?", ['<Please select>','Yes', 'No'], index=0, key='schufa_requested')


        placeholder_map = st.empty()
        with placeholder_map.container():
            st.subheader("""
                    \n
                    WG-info
                    """)
            col1, col2, col3 = st.columns(3)
            col1.select_slider(label='Female flatmates', options=[str(n) for n in range(0,11)], value='0', key='female_flatmates')
            col2.select_slider(label='Male flatmates', options=[str(n) for n in range(0,11)], value='0', key='male_flatmates')
            col3.select_slider(label='Diverse flatmates', options=[str(n) for n in range(0,11)], value='0', key='diverse_flatmates')
            col1.select_slider(label='Flatmates min age', options=[str(n) for n in range(0,100)], value='0', key='min_age_flatmates')
            col2.select_slider(label='Flatmates max age', options=[str(n) for n in range(0,100)], value='0', key='max_age_flatmates')
            col3.selectbox("Smoking", ['<Please select>','Allowed everywhere', 'Allowed in your room', 'Allowed on the balcony', 'No smoking'], index=0, key='smoking')

            placeholder_map = st.empty()
            with placeholder_map.container():
                col1, col2 = st.columns(2)
                col1.multiselect(label='Type of WG', options=['<Please select>','WG types'], default=None, key='wg_type')
                col2.multiselect(label='Spoken languages', options=['<Please select>','Languages'], default=None, key='languages')

        placeholder_map = st.empty()
        with placeholder_map.container():
            st.subheader("""
                    \n
                    Energy and power
                    """)
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.selectbox(label='Certification type', options=['<Please select>','Requirement','Consumption'], index=0, key='energy_certificate')
            col2.text_input("Power (in kW h/(m²a))", value="", key='kennwert', max_chars = 20)
            col3.multiselect(label='Heating energy source', options=['Energy sources'], default=None, key='energy_source')
            col4.text_input(label='Building construction year', value="", key='building_year', max_chars = 20)
            col5.multiselect(label='Energy efficiency class', options=['A+','A','B','C','D','E','F','G','H'], default=None, key='energy_efficiency')



        placeholder_map = st.empty()
        with placeholder_map.container():
            st.subheader("""
                    \n
                    Utils
                    """)
            col1, col2, col3 = st.columns(3)
            col1.selectbox(label='Heating', options=['<Please select>','Central heating','Gas heating', 'Furnace heating', 'District heating', 'Coal oven', 'Night storage heating'], index=0, key='heating')
            col1.multiselect(label='Internet', options=['DSL', 'Flatrate', 'WLAN'], default=None, key='internet')
            col1.selectbox(label='Internet speed', options=['<Please select>','Slower than 10 Mbit/s','Up to 10 Mbit/s','Up to 16 Mbit/s','Up to 25 Mbit/s','Up to 50 Mbit/s','Up to 100 Mbit/s','Faster than 100 Mbit/s'], index=0, key='internet_speed')
            col2.multiselect(label='Furniture', options=['Furnished', 'Partly furnished'], default=None, key='furniture')
            col2.multiselect(label='Floor type', options=['Floor boards', 'Parquet', 'Laminate', 'Carpet', 'Tiles', 'PVC', 'Underfloor heating'], default=None, key='floor_type')
            col3.multiselect(label='TV', options=['Cable', 'Satellite'], default=None, key='tv')
            col3.multiselect(label='Miscellaneous', options=['Washing machine', 'Dishwasher', 'Terrace', 'Balcony', 'Garden', 'Shared garden', 'Basement', 'Elevator', 'Pets allowed', 'Bicycle storage'], default=None, key='extras')

        "---"
        submitted_form = st.form_submit_button("Submit form")




        if submitted_form:
            full_address = str(st.session_state.address) + ', ' + str(st.session_state.zip_code) + ', ' + str(st.session_state.city)

            lat, lon = geocoding_address(full_address)

            detail_dict = {
            'id': [''],
            'url': [''],
            'type_offer': ['WG'],
            'landlord_type': ['Private'],
            'title': [''],
            'price_euros': [np.nan],
            'size_sqm': [int(st.session_state.room_size)],
            'available_rooms': [1],
            'WG_size': [int(st.session_state.total_flat_size)],
            'available_spots_wg': [1],
            'male_flatmates': [int(st.session_state.male_flatmates)],
            'female_flatmates': [int(st.session_state.female_flatmates)],
            'diverse_flatmates': [int(st.session_state.diverse_flatmates)],
            'published_on': [str(time.strftime(f"%d.%m.%Y", time.localtime()))],
            'published_at': [int(time.strftime(f"%H", time.localtime()))],
            'address': [full_address],
            'city': [str(st.session_state.city)],
            'crawler': ['WG-Gesucht'],
            'latitude': [float(lat)],
            'longitude': [float(lon)],
            'available from': [str(time.strftime(f"%d.%m.%Y", time.localtime()))],
            'available to': [np.nan]
                }


            # building_type
            # floor
            # parking
            # distance_public_transport
            # barrier_free
            # schufa_requested
            # min_age_flatmates
            # max_age_flatmates
            # smoking
            # wg_type
            # languages
            # energy_certificate
            # energy_source
            # building_year
            # energy_efficiency
            # heating
            # internet
            # furniture
            # floor_type
            # extras

            st.markdown("""
                    ### Page under construction.
                    """, unsafe_allow_html=True)

with st.expander("I found an ad in wg-gesucht.de and want to know if the price follows the market"):
    with st.form("entry_url", clear_on_submit=False):
        st.text_input("Paste here the link to a wg-gesucht.de ad. The link should look like this: 'https://www.wg-gesucht.de/wg-zimmer-in-City.1234567.html'", value="", key='url', max_chars = 250)

        "---"
        submitted_url = st.form_submit_button("Submit url")

        if submitted_url:
            url = st.session_state["url"]
            url_ok = False
            if url == '':
                st.markdown("""
                    Please submit the link to the page of the ad of your interest.
                    """, unsafe_allow_html=True)
                url_ok = False
            elif 'wg-gesucht.de' in url:
                url_ok = True
            else:
                st.markdown("""
                    The link must be from wg-gesucht.de
                    """, unsafe_allow_html=True)


            if url_ok:
                st.markdown(f'Analysing {url}', unsafe_allow_html=True)

                ## Process url to obtain table for prediction
                ad_df = crawl_ind_ad_page2(url)

                try:
                    ad_df_processed = process_ads_tables(input_ads_df = ad_df, save_processed = False, df_feats_tag = 'city')

                    ## Load model for prediction (locally or from Github)
                    # trained_model = pickle.load(open(f'{ROOT_DIR}/model/PredPipeline_WG_allcities_price_per_sqm_cold.pkl','rb'))
                    with open('https://github.com/chvieira2/wg_price_predictor/blob/main/wg_price_predictor/models/PredPipeline_WG_allcities_price_per_sqm_cold.pkl', 'rb') as f:
                        trained_model = pickle.load(f)

                    ## Make predictions
                    pred_price_sqm = float(trained_model.predict(ad_df_processed))
                    cold_rent_pred = round(float(pred_price_sqm*ad_df_processed['size_sqm']),2)
                    warm_rent_pred =  round(float(cold_rent_pred + ad_df_processed['mandatory_costs_euros'] + ad_df_processed['extra_costs_euros']),2)


                    ## Ad evaluation
                    ad_evaluation = 'over' if float(ad_df_processed['price_euros']) > warm_rent_pred*1.2 else 'under' if float(ad_df_processed['price_euros']) < warm_rent_pred*1.2 else 'fair'


                    if ad_evaluation == 'over':
                        ad_evaluation = f"***OVERPRICED***. Even after taking the margin of error in consideration, the rent price in this ad ({round(float(ad_df_processed['price_euros']),2)}€) is significantly above the price predicted by our model"
                    elif ad_evaluation == 'fair':
                        ad_evaluation = f"***FAIRLY PRICED***. Taking the margin of error in consideration, the rent price in this ad ({round(float(ad_df_processed['price_euros']),2)}€) is within the price range predicted by our model"
                    elif ad_evaluation == 'under':
                        ad_evaluation = f"***UNDERPRICED***. Even after taking the margin of error in consideration, the rent price in this ad ({round(float(ad_df_processed['price_euros']),2)}€) is significantly below the price predicted by our model"

                    # Display predictions
                    st.markdown(f"""
                                The predicted rent for this offer is: ***{warm_rent_pred}***€. This prediction is composed of the predicted cold rent ({cold_rent_pred}€), plus mandatory and extra costs indicated in the ad.

                                The offer in this ad is priced at {round(float(ad_df_processed['price_euros']),2)}€ (warm rent). We consider this price to be {ad_evaluation}.
                                """, unsafe_allow_html=True)


                except Exception as err:
                    st.markdown(f"""
                            The analysis failed. Most common reasons for analysis to fail are wrong entries in the origianl ad itself. Examples of entries in the ad that lead to failed analysis:
                            - Rent price too low or too high
                            - Room size is unrealistically large/small
                            - Invalid entries
                            """, unsafe_allow_html=True)
                    print(f"Unexpected {err=}, {type(err)=}")
                    raise




            else:
                st.markdown("There was a problem connecting to the provided link. The url should look similar to this: 'https://www.wg-gesucht.de/wg-zimmer-in-City.1234567.html'", unsafe_allow_html=True)
