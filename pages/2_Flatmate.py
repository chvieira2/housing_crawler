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
            col2.text_input("Street and house number*", value="", key='address', max_chars = 100)
            col3.text_input("Neighborhood*", value="", key='neighborhood', max_chars = 100)
            col4.text_input("Zip code*", value="", key='zip_code', max_chars = 20)

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
            col3.multiselect(label='Miscellaneous', options=['Washing machine', 'Dishwasher', 'Terrace', 'Balcony', 'Garden', 'Shared garden', 'Basement', 'Elevator', 'Pets allowed', 'Bicycle storage'], default=None, key='miscellaneous')

        "---"
        submitted_form = st.form_submit_button("Submit form")

        if submitted_form:
            st.markdown("""
                    ### Page under construction.
                    """, unsafe_allow_html=True)

with st.expander("I found an ad in wg-gesucht.de and want to know if the price follows the market"):
    with st.form("entry_url", clear_on_submit=False):
        st.text_input("Paste here the link to a wg-gesucht.de ad. The link should look like this: 'https://www.wg-gesucht.de/wg-zimmer-in-City-Neighborhood.1234567.html'", value="", key='url', max_chars = 250)

        "---"
        submitted_url = st.form_submit_button("Submit url")

        if submitted_url:
            url = st.session_state["url"]
            url_ok = False
            if url == '':
                st.markdown("""
                    Please submit a link to the page of the ad of interest.
                    """, unsafe_allow_html=True)
                url_ok = False
            if ~url.startswith('https://www.'):
                url = 'https://www.' + url
                url_ok = True
            elif ~url.startswith('https://'):
                url = 'https://' + url
                url_ok = True
            if 'wg-gesucht.de' not in url:
                st.markdown("""
                    The link must be from wg-gesucht.de
                    """, unsafe_allow_html=True)
                url_ok = False

            if url_ok:
                st.text(f'Searching for {url}')
                st.markdown("""
                        ### Page under construction.
                        """, unsafe_allow_html=True)




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
#     form.selectbox(label = 'Please select a city of interest', key='input_city', options = preloaded_cities, index=preloaded_cities.index('Berlin'))
#     # form.text_input(label='Type city name', key='input_city', type="default", on_change=None, placeholder='p.ex. Berlin')


#     expander_weights = form.expander("Options")

#     # Weights selection
#     expander_weights.select_slider(label='Activities and Services:', options=list(weight_dict.keys()), value='Average', key='weight_activity')
#     expander_weights.select_slider(label='Comfort:', options=list(weight_dict.keys()), value='Average', key='weight_comfort')
#     expander_weights.select_slider(label='Mobility:', options=list(weight_dict.keys()), value='Average', key='weight_mobility')
#     expander_weights.select_slider(label='Social life:', options=list(weight_dict.keys()), value='Average', key='weight_social')



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
