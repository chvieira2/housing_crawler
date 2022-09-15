# -*- coding: utf-8 -*-

"""Housing Crawler - search for flats by crawling property portals and save them locally.

This is the dashboard/app implementation of the analysis of ads obtained from wg-gesucht.de"""

__author__ = "Carlos Henrique Vieira e Vieira"
__version__ = "1.0"
__maintainer__ = "chvieira2"
__email__ = "carloshvieira2@gmail.com"
__status__ = "Production"


# from sqlalchemy import null
import time
import streamlit as st
import calendar  # Core Python Module
import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd

import plotly.graph_objects as go  # pip install plotly
from streamlit_option_menu import option_menu  # pip install streamlit-option-menu

from housing_crawler.ads_table_processing import get_processed_ads_table
from housing_crawler.params import dict_city_number_wggesucht

# -------------- SETTINGS --------------
target='price_per_sqm_cold' # price_per_sqm_cold, price_euros
target_log_transform = False
# --------------------------------------


#-----------------------page configuration-------------------------
st.set_page_config(
    page_title="housing_crawler",
    page_icon=':house:', # gives a random emoji //to be addressed for final ver
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


#----Page start------
placeholder_map = st.empty()

with placeholder_map.container():

    st.markdown("""
                # Welcome to the <span style="color:tomato">WG price Germany</span> dashboard!
                ## Here you'll find everything you want to know about WG prices in Germany.
                For technical information, please check my [GitHub](https://github.com/chvieira2/housing_crawler).
                """, unsafe_allow_html=True)


#     # --- NAVIGATION MENU ---
#     selected = option_menu(
#         menu_title=None,
#         options=["General info", "Cities ranking"],
#         icons=["bar-chart-fill", "list-ol"],  # https://icons.getbootstrap.com/
#         orientation="horizontal",
# )

# --- General info ---
# if selected == "General info":
    with st.form("entry_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        col1.selectbox("Analysis period:", ['one week','one month', 'three months', 'six months', 'one year'], key="period", index=1)
        col2.selectbox("City:", ['All cities'] + list(dict_city_number_wggesucht.keys()), key="city", index=0)

        "---"
        submitted = st.form_submit_button("Show results")

        if submitted:
            #### Obtain main ads table
            ads_df = get_processed_ads_table()

            ## Format dates properly
            ads_df['published_on'] = pd.to_datetime(ads_df['published_on'], format = "%Y-%m-%d")

            ## Filter table
            # City of choice
            if st.session_state["city"] != 'All cities':
                ads_df = ads_df[ads_df['city'] == str(st.session_state["city"])]


            # Filter ads in between desired dates.
            date_max = pd.to_datetime(time.strftime("%Y-%m-%d", time.localtime()), format = "%Y-%m-%d")

            if st.session_state["period"] == 'one week':
                date_min = datetime.date.today() + relativedelta(weeks=-1)
                date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
            elif st.session_state["period"] == 'one month':
                date_min = datetime.date.today() + relativedelta(months=-1)
                date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
            elif st.session_state["period"] == 'three months':
                date_min = datetime.date.today() + relativedelta(months=-3)
                date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
            elif st.session_state["period"] == 'six months':
                date_min = datetime.date.today() + relativedelta(months=-6)
                date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
            elif st.session_state["period"] == 'one year':
                date_min = datetime.date.today() + relativedelta(months=-12)
                date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
            else:
                date_min = datetime.date.today() + relativedelta(months=-48)
                date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")

            ads_df['temp_col'] = ads_df['published_on'].apply(lambda x: x >= date_min and x <= date_max)
            ads_df = ads_df[ads_df['temp_col']].drop(columns=['temp_col'])







            ## Filter type of offer
            wg_df = ads_df.query('type_offer_simple == "WG"').reset_index().drop(columns=['index'])

            singleroom_df = ads_df.query('type_offer_simple == "Single-room flat"').reset_index().drop(columns=['index'])

            flathouse_df = ads_df.query('(type_offer_simple == "Apartment")').reset_index().drop(columns=['index'])


            # period = str(st.session_state["year"]) + "_" + str(st.session_state["month"])
            # incomes = {income: st.session_state[income] for income in incomes}
            # expenses = {expense: st.session_state[expense] for expense in expenses}
            # db.insert_period(period, incomes, expenses, comment)
            # st.success("Data saved!")














# #------------------------user inputs-----------------------------------
# #inputs for weights for users
# weight_dict={"Don't care much":1/10,
#              "Somewhat important":1/3,
#              'Average':1,
#              'Quite important':3,
#              'Very important':10}

# user_number_pages_dict={"few":1,
#              "some":3,
#              'many':5}

# with st.sidebar:
#     form = st.form("calc_weights")

#     # City selection
#     form.selectbox(label = 'Select a city of interest', key='input_city', options = preloaded_cities, index=preloaded_cities.index('Berlin'))
#     # form.text_input(label='Type city name', key='input_city', type="default", on_change=None, placeholder='p.ex. Berlin')


#     expander_weights = form.expander("Options")

#     # Weights selection
#     expander_weights.select_slider(label='Activities and Services:', options=list(weight_dict.keys()), value='Average', key='weight_activity', help=None, on_change=None)
#     expander_weights.select_slider(label='Comfort:', options=list(weight_dict.keys()), value='Average', key='weight_comfort', help=None, on_change=None)
#     expander_weights.select_slider(label='Mobility:', options=list(weight_dict.keys()), value='Average', key='weight_mobility', help=None, on_change=None)
#     expander_weights.select_slider(label='Social life:', options=list(weight_dict.keys()), value='Average', key='weight_social', help=None, on_change=None)



#     ## Checkbox for wg-gesuch ads
#     expander = form.expander("Housing offers (German cities only)")

#     cbox_wggesucht = expander.checkbox('Display housing offers?')

#     # Search filter criterium
#     user_filters = expander.multiselect(
#                 'Search filters',
#                 ["Room in flatshare","Single room flat","Flat","House"],
#                 ["Room in flatshare"])

#     dict_filters = {"Room in flatshare":"wg-zimmer",
#                     "Single room flat":"1-zimmer-wohnungen",
#                     "Flat":"wohnungen",
#                     "House":"haeuser"}
#     user_filters = [dict_filters.get(filter) for filter in user_filters]

#     # Number of search pages
#     user_number_pages = expander.radio('Number of housing offers to display', user_number_pages_dict.keys())

#     user_number_pages = user_number_pages_dict.get(user_number_pages)

#     #Form submit button to generate the inputs from the user
#     submitted = form.form_submit_button('Display livability map', on_click=None)


# ## Page after submission
# if submitted:
#     placeholder_map.empty()
#     weights_inputs = (st.session_state.weight_activity,
#                st.session_state.weight_comfort,
#                st.session_state.weight_mobility,
#                st.session_state.weight_social,
#             #    'Average' # Last weight 'average' refers to negative features
#                )
#     weights=[weight_dict[i] for i in weights_inputs]
#     #check weights
#     print(f'Weights entered by user: {weights}')

#     city = LivabilityMap(location=st.session_state.input_city, weights=weights)
#     city.calc_livability()
#     df_liv = city.df_grid_Livability

#     # MinMax scale all columns for display
#     categories_interest = ['activities_mean', 'comfort_mean', 'mobility_mean', 'social_mean']
#     df_liv = min_max_scaler(df_liv, columns = categories_interest)
#     #city center position lat,lon
#     # city_coords = [np.mean(df_liv['lat_center']),np.mean(df_liv['lng_center'])]
#     row_max_liv = df_liv[df_liv['livability'] == max(df_liv['livability'])].head(1)
#     city_coords = [float(row_max_liv['lat_center']),float(row_max_liv['lng_center'])]

#     print(f"""============ {city.location} coordinates: {city_coords} =============""")


#     #for filling the polygon
#     style_function = {'fillColor': 'transparent',
#                  'lineColor': '#00FFFFFF'}
#     # city borders map
#     geojson_path=f'{ROOT_DIR}/housing_crawler/data/{city.location_name}/{city.location_name}_boundaries.geojson'
#     file = open(geojson_path, encoding="utf8").read()
#     city_borders = GeoJson(file,
#                           name=city.location,
#                           show=True,
#                           style_function=lambda x:style_function,
#                           zoom_on_click=True)
#     mapObj = plot_map(df_liv, city_coords, city_borders)
#     #Used to fill the placeholder of the world map with according one of the selected city
#     with placeholder_map.container():
#         st.markdown(f"""
#                 # Here's the livability map for <span style="color:tomato">{city.location}</span><br>
#                 """, unsafe_allow_html=True)

#         displayed_map = st.empty()
#         with displayed_map:
#             stf.folium_static(mapObj, width=700, height=500)
#         st.markdown(f"""
#             The livability score is only as good as the data available for calculating it. Be aware that the sourced data from OpenStreetMap that is used for calculating the livability score differs in quality around the world.
#             """)

#         ## Add wg-gesucht ads
#         if city.location in list(dict_city_number_wggesucht.keys()) and cbox_wggesucht:
#             start_placeholder = st.empty()
#             start_placeholder.markdown("""
#                     Searching for housing offers...<br>
#                     If this is taking longer than 3-5 minutes, please try again later.
#                     """, unsafe_allow_html=True)

#             # Obtain recent ads
#             CrawlWgGesucht().crawl_all_pages(location_name = city.location,
#                                              number_pages = user_number_pages,
#                                              filters = user_filters)

#             df = pd.read_csv(f"{ROOT_DIR}/housing_crawler/data/{standardize_characters(city.location)}/Ads/{standardize_characters(city.location)}_ads.csv")
#             print(f'===> Loaded ads')

#             ## Filter ads table
#             # Remove ads without latitude and longitude
#             df = df.dropna(subset=['latitude'])

#             ## Add ads to map
#             for index,row in df.iterrows():
#                 livability = get_liv_from_coord(row.loc['latitude'],row.loc['longitude'],liv_df = df_liv)
#                 if 'WG' in row.loc['type_offer']:
#                     tooltip = f"""
#                               {row.loc['title']}<br>
#                               Address: {row.loc['address']}<br>
#                               Rent (month): {row.loc['price_euros']} €<br>
#                               Room size: {row.loc['size_sqm']} sqm<br>
#                               Capacity: {row.loc['WG_size']} people<br>
#                               Published on: {row.loc['published_on']}<br>
#                               Available from: {'' if pd.isnull(row.loc['available_from']) else row.loc['available_from']}<br>
#                               Available until: {'' if pd.isnull(row.loc['available_to']) else row.loc['available_to']}<br>
#                               Location livability score: {int(0 if pd.isnull(livability) else 100*livability)}%
#                               """
#                 elif '1 Zimmer Wohnung' in row.loc['type_offer']:
#                     tooltip = f"""
#                               {row.loc['title']}<br>
#                               Address: {row.loc['address']}<br>
#                               Rent (month): {row.loc['price_euros']} €<br>
#                               Property size: {row.loc['size_sqm']} sqm<br>
#                               Published on: {row.loc['published_on']}<br>
#                               Available from: {row.loc['available_from']}<br>
#                               Available until: {'open end' if pd.isnull(row.loc['available_to']) else row.loc['available_to']}<br>
#                               Location livability score: {int(0 if pd.isnull(livability) else 100*livability)}%
#                               """
#                 else:
#                     tooltip = f"""
#                               {row.loc['title']}<br>
#                               Address: {row.loc['address']}<br>
#                               Rent (month): {row.loc['price_euros']} €<br>
#                               Property size: {row.loc['size_sqm']} sqm<br>
#                               Rooms: {row.loc['available_rooms']} rooms<br>
#                               Published on: {row.loc['published_on']}<br>
#                               Available from: {row.loc['available_from']}<br>
#                               Available until: {'open end' if pd.isnull(row.loc['available_to']) else row.loc['available_to']}<br>
#                               Location livability score: {int(0 if pd.isnull(livability) else 100*livability)}%
#                               """


#                 folium.Marker(location=[row.loc['latitude'], row.loc['longitude']],
#                               tooltip=tooltip,
#                               popup=f"""
#                               <a href="{row.loc['url']}">{row.loc['url']}</a>
#                               """,
#                                 icon=folium.Icon(color='purple', icon='home'))\
#                     .add_to(mapObj)

#             ## Display map
#             start_placeholder.markdown("""
#                     Showing recently posted housing offers in your city. Be aware that the displayed locations are approximated.<br> This list is not comprehensive and more offers are available at [wg-gesucht.de](wg-gesucht.de).
#                     """, unsafe_allow_html=True)

#         with displayed_map:
#             stf.folium_static(mapObj, width=500, height=500)
