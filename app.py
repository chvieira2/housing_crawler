# -*- coding: utf-8 -*-

"""Housing Crawler - search for flats by crawling property portals and save them locally.

This is the dashboard/app implementation of the analysis of ads obtained from wg-gesucht.de"""

__author__ = "Carlos Henrique Vieira e Vieira"
__version__ = "1.0"
__maintainer__ = "chvieira2"
__email__ = "carloshvieira2@gmail.com"
__status__ = "Production"

from config.config import ROOT_DIR


import time
import streamlit as st
from streamlit_folium import st_folium
import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'. Needed to remove SettingWithCopyWarning warning when assigning new value to dataframe column
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from housing_crawler.params import dict_city_number_wggesucht
from housing_crawler.string_utils import standardize_characters
from housing_crawler.utils import crawl_ind_ad_page2, get_data, obtain_latest_model
from housing_crawler.ads_table_processing import process_ads_tables
from housing_crawler.geocoding_addresses import geocoding_address

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


# ---------------------- FUNCTIONS ----------------------
@st.cache
def get_data_from_db(file_name_tag='ads_OSM.csv', local_file_path=f'raw_data'):
    """
    Method to get data from local environment and return a unified dataframe

    """
    return get_data(file_name_tag=file_name_tag, local_file_path=local_file_path)

@st.cache
def get_latest_model_from_db():
    """
    Method to get latest trained model from local dataabse

    """
    return obtain_latest_model()

def filter_original_data(df, city, time_period):
    ## Format dates properly
    df['published_on'] = pd.to_datetime(df['published_on'], format = "%Y-%m-%d")

    ## Filter table
    # City of choice
    if st.session_state["city_filter"] != 'Germany':
        df = df[df['city'] == city]


    # Filter ads in between desired dates.
    date_max = pd.to_datetime(time.strftime("%Y-%m-%d", time.localtime()), format = "%Y-%m-%d")

    if time_period == 'Past week':
        date_min = datetime.date.today() + relativedelta(weeks=-1)
        date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
    elif time_period == 'Past month':
        date_min = datetime.date.today() + relativedelta(months=-1)
        date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
    elif time_period == 'Past three months':
        date_min = datetime.date.today() + relativedelta(months=-3)
        date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
    elif time_period == 'Past six months':
        date_min = datetime.date.today() + relativedelta(months=-6)
        date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
    elif time_period == 'Past year':
        date_min = datetime.date.today() + relativedelta(months=-12)
        date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
    else:
        date_min = datetime.date.today() + relativedelta(months=-48)
        date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")

    df['temp_col'] = df['published_on'].apply(lambda x: x >= date_min and x <= date_max)
    df = df[df['temp_col']].drop(columns=['temp_col'])


    ### Correcting Zip format
    df['zip_code'] = df['zip_code'].apply(lambda x: 'No address' if x != x else str(int(x)))

    return df

def ads_per_region_stacked_barplot(df,time_period, city):

    if city != 'Germany':
        stacking_by = 'zip_code'
        st.markdown(f'Ads published on wg-gesucht.de in {city} in the {time_period.lower()}.', unsafe_allow_html=True)
    else:
        stacking_by = 'city'
        st.markdown(f'Ads published on wg-gesucht.de in the selected 25 cities in Germany in the {time_period.lower()}.', unsafe_allow_html=True)



    region_ads_df = df[['url', stacking_by,"type_offer_simple"]].groupby([stacking_by,"type_offer_simple"]).count().rename(columns = {'url':'count'}).sort_values(by = ['count'], ascending=False).reset_index()


    fig = px.bar(region_ads_df, x=stacking_by, y="count", color="type_offer_simple",
            labels={
                stacking_by: "City" if stacking_by == 'city' else 'Zip code',
                "count": 'Number of ads published',
                "type_offer_simple": "Type of ad"
            },
            template = "ggplot2" #["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
            )
    fig.update_layout(height=300,
                        font_family="Arial",
                        margin= dict(
                        l = 10,        # left
                        r = 10,        # right
                        t = 25,        # top
                        b = 0,        # bottom
                ))
    if city != 'Germany':
        fig.update_layout(xaxis_title='Zip code')
    else:
        fig.update_layout(xaxis_title=None)

    return fig

def ads_per_day_stacked_barplot(df,city,time_period,market_type):

    if city != 'Germany':
        stacking_by = 'zip_code'
    else:
        stacking_by = 'city'

    df_plot = df[['url', stacking_by, 'published_on']].groupby(['published_on',stacking_by]).count().rename(columns={'url':'count'}).sort_values(by = ['published_on'], ascending=True).reset_index()

    foo = df_plot[['count', stacking_by, 'published_on']].groupby(['published_on']).sum().rename(columns={'count':'mean'})
    mean_ads_day = int(round(foo['mean'].mean(),0))

    st.markdown(f'On average {mean_ads_day} {market_type} ads were published on wg-gesucht.de every day in {city} in the {time_period.lower()}.', unsafe_allow_html=True)

    fig = px.bar(df_plot, x="published_on", y="count", color=stacking_by,
                    # title=' ',
            labels={
                "published_on": 'Published on',
                "count": 'Number of ads published per day',
                stacking_by: "City" if stacking_by == 'city' else 'Zip code'
            },
            template = "seaborn" #["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
            )
    fig.update_layout(height=300,
                        xaxis_title=None,
                        font_family="Arial",
                        margin= dict(
                        l = 10,        # left
                        r = 10,        # right
                        t = 25,        # top
                        b = 0,        # bottom
                ))
    fig.add_shape( # add a horizontal "target" line
    type="line", line_color="darkgrey", line_width=3, opacity=1, line_dash="dot",
    x0=0, x1=1, xref="paper", y0=mean_ads_day, y1=mean_ads_day, yref="y"
                )
    return fig

def ads_per_hour_line_polar(df,city,time_period,market_type):
    df_time = df[['url', 'day_of_week_publication','published_at']].groupby(['day_of_week_publication','published_at']).count().rename(columns = {'url':'count'}).reset_index()

    plotting_values = [float(i) for i in list(np.arange(0, 360, 360/24))]
    mapping_dict = dict(zip(range(0,24), plotting_values))
    df_time['published_at_radians'] = df_time.published_at.map(mapping_dict)

    days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    st.markdown(f'Average number of {market_type} ads published per hour on each day of the week in {city} in the {time_period.lower()}.', unsafe_allow_html=True)

    fig = px.line_polar(df_time,
                        r="count",
                        theta="published_at_radians",
                        color="day_of_week_publication",
                        labels={
                            "count": 'Number of ads published per hour',
                            "published_at_radians": 'Hour of the day',
                            "day_of_week_publication": ""
                        },
                        category_orders={"day_of_week_publication": days_of_week},
                        line_close=True,
                        template="seaborn")

    fig.update_layout(height=250,
                        # width=25,
                        font_family="Arial",
                        margin= dict(
                        l = 30,        # left
                        r = 30,        # right
                        t = 20,        # top
                        b = 0,        # bottom
                        ),
                        legend_orientation='h',
                        # legend_y=1,
                        legend_x=0)
    fig.update_polars(angularaxis_tickmode='array',
                      angularaxis_tickvals=list(np.arange(0, 360, 360/24)),
                      angularaxis_ticktext=[str(val)+'h' for val in list(range(0,24))],
                      radialaxis_angle=45,
                    #   radialaxis_type='linear',
                    #   radialaxis_tickvals=[50,100,150],
                    #   radialaxis_range=[0,200]
                      )

    return fig

def price_rank_cities(df,city):

    if city != 'Germany':
        stacking_by = 'zip_code'
    else:
        stacking_by = 'city'

    # Log transform prices
    wg_df_mod = df.query('type_offer_simple == "WG"')
    wg_df_mod['price_euros'] = np.log2(wg_df_mod['price_euros'])

    singleroom_df_mod = df.query('type_offer_simple == "Single-room flat"')
    singleroom_df_mod['price_euros'] = np.log2(singleroom_df_mod['price_euros'])

    flathouse_df_mod = df.query('type_offer_simple == "Apartment"')
    flathouse_df_mod['price_euros'] = np.log2(flathouse_df_mod['price_euros'])

    # Finding the order
    city_wg_df = wg_df_mod[['url', 'price_euros', stacking_by]].groupby(stacking_by).agg({'price_euros': 'mean','url': 'count'}).sort_values(by = ['price_euros'], ascending=False).rename(columns = {'url':'count'})
    city_singleroom_df = singleroom_df_mod[['url', 'price_euros', stacking_by]].groupby(stacking_by).agg({'price_euros': 'mean','url': 'count'}).sort_values(by = ['price_euros'], ascending=False).rename(columns = {'url':'count'})
    city_flathouse_df = flathouse_df_mod[['url', 'price_euros', stacking_by]].groupby(stacking_by).agg({'price_euros': 'mean','url': 'count'}).sort_values(by = ['price_euros'], ascending=False).rename(columns = {'url':'count'})


    # Filter table for groups with at least 3 entries
    city_wg_df = city_wg_df[city_wg_df['count']>= 3]
    city_singleroom_df = city_singleroom_df[city_singleroom_df['count']>= 3]
    city_flathouse_df = city_flathouse_df[city_flathouse_df['count']>= 3]

    # Figure
    sns.set_theme(style = "whitegrid", font_scale= 1.5)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15,len(city_wg_df)*0.3))

    sns.boxenplot(data = wg_df_mod, y = stacking_by, x = 'price_euros',
                order = city_wg_df.index,
                palette='crest', showfliers = False,
                ax=ax1).set(title='WGs')

    sns.boxenplot(data = singleroom_df_mod, y = stacking_by, x = 'price_euros',
                order = city_singleroom_df.index,
                palette='crest', showfliers = False,
                ax=ax2).set(title='Single-room flat')

    sns.boxenplot(data = flathouse_df_mod, y = stacking_by, x = 'price_euros',
                order = city_flathouse_df.index,
                palette='crest', showfliers = False,
                ax=ax3).set(title='Multiple-rooms flat')

    ax1.set(xlim=(7, 12), xlabel=None, ylabel=None)
    ax1.set_xticks([np.log2(250),np.log2(500),np.log2(1000),np.log2(2000),np.log2(4000)])
    ax1.set_xticklabels(rotation = 90, labels =[250,500,1000,2000,4000])


    ax2.set(xlim=(7, 12), xlabel=None, ylabel=None)
    ax2.set_xticks([np.log2(250),np.log2(500),np.log2(1000),np.log2(2000),np.log2(4000)])
    ax2.set_xticklabels(rotation = 90, labels =[250,500,1000,2000,4000])

    ax3.set(xlim=(7, 12), xlabel=None, ylabel=None)
    ax3.set_xticks([np.log2(250),np.log2(500),np.log2(1000),np.log2(2000),np.log2(4000)])
    ax3.set_xticklabels(rotation = 90, labels =[250,500,1000,2000,4000])

    plt.tight_layout()



    return fig

def prepare_data_for_map(ads_df):
    # This part of the code was adapted from: https://juanitorduz.github.io/germany_plots/

    ## Obtain Germany info summary
    import geopandas as gpd
    germany_df = gpd.read_file(f'{ROOT_DIR}/housing_crawler/data/germany_summary.shp', dtype={'plz': str})
    # print(sorted(list(set(germany_df['ort']))))

    germany_df = germany_df[germany_df['ort'].isin(list(dict_city_number_wggesucht.keys()))]
    germany_df['zip_code'] = germany_df['plz'].astype('int64')
    # Filter out PLZ 86692 that ends up in the data because there is a second city called Münster in Bavaria
    germany_df = germany_df[germany_df['zip_code'] != 86692]


   ## Prepare dataframes for merging with Country information
    def prepare_df_for_merge_plz(df, tag):

        # Mean price per PLZ
        df_plz = df[['price_per_sqm_cold', 'zip_code']].groupby(['zip_code']).mean().reset_index()
        df_plz['zip_code'] = df_plz['zip_code'].astype('int64')
        df_plz['price_per_sqm_cold'] = round(df_plz['price_per_sqm_cold'],2)
        df_plz = df_plz.rename(columns = {'price_per_sqm_cold':f'price_per_sqm_{tag}'})

        # SD price per PLZ
        df_plz_std = df[['price_per_sqm_cold', 'zip_code']].groupby(['zip_code']).std().reset_index()
        df_plz_std['price_per_sqm_cold'] = round(df_plz_std['price_per_sqm_cold'],1)
        df_plz_std = df_plz_std.rename(columns = {'price_per_sqm_cold':f'std_{tag}'})
        df_plz = pd.merge(left=df_plz, right=df_plz_std, on='zip_code', how='left')

        # Count of ads per PLZ
        df_plz_count = df[['price_per_sqm_cold', 'zip_code']].groupby(['zip_code']).count().reset_index()
        df_plz_count = df_plz_count.rename(columns = {'price_per_sqm_cold':f'count_{tag}'})
        df_plz = pd.merge(left=df_plz, right=df_plz_count, on='zip_code', how='left')

        return df_plz

    wg_df_plz = prepare_df_for_merge_plz(df = ads_df.query('type_offer_simple == "WG"'), tag = 'wg')
    singleroom_df_plz = prepare_df_for_merge_plz(df = ads_df.query('type_offer_simple == "Single-room flat"'), tag = 'single')
    flathouse_df_plz = prepare_df_for_merge_plz(df = ads_df.query('type_offer_simple == "Apartment"'), tag = 'multi')



    ## Sequential merging to obtain all data in a single df
    cities_df = pd.merge(left=pd.merge(left=pd.merge(left=germany_df,
                                                    right=flathouse_df_plz, on='zip_code', how='left'),
                                    right=singleroom_df_plz, on='zip_code', how='left'),
                        right=wg_df_plz, on='zip_code', how='left')


    ## Create value for coloring and remove zip_code without info so they are not plotted
    cities_df['mean_for_color'] = round(cities_df[['price_per_sqm_wg', 'price_per_sqm_single', 'price_per_sqm_multi']].mean(axis=1),2)
    # cities_df = cities_df[~cities_df['mean_for_color'].isnull()].reset_index(drop=True)
    # cities_df = cities_df['mean_for_color'].fillna(-1)


    ## Create and prepare plotting dataframe
    plotting_df = cities_df.copy()
    plotting_df['zip_code'] = plotting_df['zip_code'].astype('str')
    for tag in ['wg', 'single','multi']:
        plotting_df[tag] = plotting_df.apply(lambda x: f"{x[f'price_per_sqm_{tag}']} \u00B1{x[f'std_{tag}']} €/m² / {0 if x[f'count_{tag}'] != x[f'count_{tag}'] else int(x[f'count_{tag}'])} ads", axis=1)
    return plotting_df

def map_plotting(plotting_df, market_type):

    import folium
    from folium.plugins import Fullscreen
    import branca.colormap
    from collections import defaultdict
    from folium import GeoJson, Marker
    import geopandas as gpd

    ########### Create the map
    mapObj = folium.Map(location=(51.1657, 10.4515), zoom_start=6, max_zoom = 18, width='100%', height='100%', tiles = 'cartodbpositron')
    folium.TileLayer('openstreetmap').add_to(mapObj)


    ########### Draw borders
    # Add german states border
    feature_group = folium.FeatureGroup('States borders')
    states = ['Baden-Württemberg',
    'Bayern',
    'Berlin',
    'Brandenburg',
    'Bremen',
    'Hamburg',
    'Hessen',
    'Mecklenburg-Vorpommern',
    'Niedersachsen',
    'Nordrhein-Westfalen',
    'Rheinland-Pfalz',
    'Saarland',
    'Sachsen',
    'Sachsen-Anhalt',
    'Schleswig-Holstein',
    'Thüringen']
    for area in states:
        if area not in list(dict_city_number_wggesucht.keys()): # exclude city-estates
            feature_group.add_child(GeoJson(data = open(f'{ROOT_DIR}/housing_crawler/data/germany/bundeslaender/{standardize_characters(area)}_border.geojson', "r", encoding="utf-8-sig").read(),
                    name=area,
                    show=True,
                    style_function=lambda x:{'fillColor': 'transparent','color': 'black', 'weight':0.5},
                    zoom_on_click=False))
    feature_group.add_to(mapObj)


    # Add cities borders and markers
    feature_group = folium.FeatureGroup('City borders')
    feature_group_markers = folium.FeatureGroup('City markers')
    for area in list(dict_city_number_wggesucht.keys()):
        # City border
        feature_group.add_child(GeoJson(data = open(f'{ROOT_DIR}/housing_crawler/data/{standardize_characters(area)}/{standardize_characters(area)}_border.geojson', "r", encoding="utf-8-sig").read(),
                                        name=area,
                                        show=True,
                                        style_function=lambda x:{'fillColor': 'transparent','color': 'black', 'weight':2},
                                        zoom_on_click=True))

        # City marker
        gdp = gpd.read_file(f'{ROOT_DIR}/housing_crawler/data/{standardize_characters(area)}/{standardize_characters(area)}_border.geojson')
        centroid = gdp.to_crs('+proj=cea').centroid.to_crs(gdp.crs)
        centroid = list(centroid[0].coords)[0]
        feature_group_markers.add_child(Marker(location=[centroid[1], centroid[0]], tooltip=area,
                                        zoom_on_click=True,
    #                                    icon=folium.DivIcon(html=f"""<b>{area}</b>""",
    #                                                        class_name="mapText"),
                                    fill_color='#132b5e', num_sides=3, radius=10))
    feature_group.add_to(mapObj)
    feature_group_markers.add_to(mapObj)


    ########### PLZ color mapping
    # Filter zip codes with at least a minimum number of WG ads
    threshold_display = 3.0
    if market_type == 'WG':
        plotting_df = plotting_df[plotting_df['count_wg'] >= threshold_display]
    if market_type == 'Single-room flat':
        plotting_df = plotting_df[plotting_df['count_single'] >= threshold_display]
    if market_type == 'Apartment':
        plotting_df = plotting_df[plotting_df['count_multi'] >= threshold_display]

    colors = ['#225ea8',
            '#1d91c0',
            '#41b6c4',
            '#7fcdbb',
            '#c7e9b4',
            '#edf8b1',
            '#ffffd9'
            ]
    colormap = branca.colormap.LinearColormap(colors=colors, index= np.arange(5, 25, 20/len(colors)), vmin=5, vmax=25, caption=f'Average {market_type} rental price (€/m²)')
    colormap.to_step(1000).add_to(mapObj)

    def style_fn(feature):
        color = 'grey' if pd.isnull(feature["properties"]["price_per_sqm_wg"]) else colormap(feature["properties"]["price_per_sqm_wg"])

        ss = {
            "fillColor": color,
            'fillOpacity':0.5,
            'color': 'black',
            'weight': 0.25
        }
        return ss


    folium.GeoJson(
        plotting_df.to_json(),
        name='Rental price (€/m²)',
        show=True,
        style_function=style_fn,
        highlight_function=lambda x: {
            'fillOpacity':1
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=['ort', 'zip_code', 'wg', 'single', 'multi'],
            aliases=['City:', 'ZIP code:', 'WGs:', 'Single-room flats:', 'Multi-room flats:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
            localize=True
        ),
    ).add_to(mapObj)


    ########### Display settings
    folium.LayerControl().add_to(mapObj)
    Fullscreen(position='topleft',
            title='Full Screen',
            title_cancel='Exit Full Screen',
            force_separate_button=False).add_to(mapObj)

    # mapObj.save(outfile= "test.html")
    return mapObj

def my_boxplot(df, x, x_title, y='price_per_sqm_cold', transform_type=None,
               y_title = "€/m²", market_type = 'WG', x_axis_rotation = None,fig_height=10, font_scale= 5, order=None):
    if transform_type == 'int':
        df[x] = df[x].astype(int)
    if transform_type == 'str':
        df[x] = df[x].astype(str)
    if transform_type == 'float':
        df[x] = df[x].astype(float)


    # Log transform prices
    wg_df_mod = df.query('type_offer_simple == "WG"')
    wg_df_mod['price_euros'] = np.log2(wg_df_mod['price_euros'])

    singleroom_df_mod = df.query('type_offer_simple == "Single-room flat"')
    singleroom_df_mod['price_euros'] = np.log2(singleroom_df_mod['price_euros'])

    flathouse_df_mod = df.query('type_offer_simple == "Apartment"')
    flathouse_df_mod['price_euros'] = np.log2(flathouse_df_mod['price_euros'])



    if order == 'mean':
        grouped = wg_df_mod[[x,y]].groupby(x).mean(y).sort_values(by=y,ascending=True)
        order = grouped.index


    # Start figure
    sns.set_theme(style = "whitegrid", font_scale= font_scale)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, fig_height))

    if market_type == 'WG':
        sns.boxenplot(data = wg_df_mod,
                        x = x, y = y,
                        palette='Blues',
                        order=order,
                        showfliers = False,
                        ax=ax)
    else:
        sns.boxenplot(data = df,
                        x = x, y = y,
                        palette='Blues',
                        hue='type_offer_simple',
                        showfliers = False,
                        ax=ax)

        ax.legend(title='', fontsize=50, bbox_to_anchor= (0.725,1.5))# loc='upper right')


    plt.ylabel(y_title)
    plt.xlabel(x_title)
    if x_axis_rotation is not None:
        # plt.setp(ax.get_xticklabels(), rotation=x_axis_rotation)
        plt.xticks(rotation='vertical')

    return fig


# ---------------------- PAGE START ----------------------
st.markdown("""
                # Welcome to <span style="color:tomato">WG-prices</span>!
                """, unsafe_allow_html=True)
#
#     # --- NAVIGATION MENU ---
#     selected = option_menu(
#         menu_title=None,
#         options=["General info", "Cities ranking"],
#         icons=["bar-chart-fill", "list-ol"],  # https://icons.getbootstrap.com/
#         orientation="horizontal",
# )




###############################################
### Creates the different tabs with results ###
###############################################
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analyse WG from url link", "Analyse my own WG",
                                        "Overview of the WG market",
                                        'The predictive model of WG prices', 'About'])

with tab1:
    st.markdown("""
            ### Check how a offer compares to thousands of other ads from [WG-gesucht](https://www.wg-gesucht.de/).
            """, unsafe_allow_html=True)

    with st.form("entry_url", clear_on_submit=False):
        st.text_input("Your wg-gesucht link:", value="", key='url', max_chars = 250)

        submitted_url = st.form_submit_button("Submit url")




        if submitted_url:


            #### Preparing for analysis ####
            url = str(st.session_state["url"])
            url_ok = False
            if url == '' or url == 'https://www.wg-gesucht.de/wg-zimmer-in-City-Neighborhood.1234567.html' or not url.startswith('https://www.wg-gesucht.de'):
                st.markdown("""
                    Please submit a valid link to a WG ad from wg-gesucht.de.

                    A valid link has the format "https://www.wg-gesucht.de/wg-zimmer-in-City-Neighborhood.1234567.html", where the Neighborhood tag may or may not be present.
                    """, unsafe_allow_html=True)

            elif not url.startswith('https://www.wg-gesucht.de/wg-zimmer-'):
                st.markdown("""
                    The ad is not a WG. At the moment only WGs are supported.
                    """, unsafe_allow_html=True)

            else:
                url_ok = True


            if url_ok:
                st.markdown(f"""
                            Analysing {url}. This could take a minute.
                            """, unsafe_allow_html=True)


                ## Process url to obtain table for prediction
                ad_df = crawl_ind_ad_page2(url)

                ad_df_processed = None
                try:
                    ## Process ad_df for analysis
                    ad_df_processed = process_ads_tables(input_ads_df = ad_df, save_processed = False, df_feats_tag = 'city')

                except Exception as err:
                    st.markdown(f"""
                            The analysis failed. Most common reasons for analysis to fail are wrong entries in the original ad in wg-gesucht.de.

                            Examples of entries in the ad that lead to failed analysis:
                            - Rent price too low or too high
                            - Room size is unrealistically large/small
                            - Invalid entries in other fields

                            Please get in contact if you think none of these reasons apply in your case.
                            """, unsafe_allow_html=True)
                    print(f"Unexpected {err=}, {type(err)=}")
                    raise



                if ad_df_processed is not None:

            #### Collect files needed for analysis ####
                    ### Obtain main ads table ###
                    # Copying is needed to prevent subsequent steps from modifying the cached result from get_original_data()
                    ads_df = get_data_from_db().copy()
                    ### Filter data for analysis ###
                    ## WGs only
                    ads_df = ads_df[ads_df['type_offer_simple'] == 'WG']

                    ## Remove ad of interest from database
                    ads_df = ads_df[ads_df['id'] != list(ad_df_processed['id'])[0]]


                    ## Filter for past 3 months only
                    # Format dates properly
                    ads_df['published_on'] = pd.to_datetime(ads_df['published_on'], format = "%Y-%m-%d")
                    date_three_months_ago = datetime.date.today() + relativedelta(months=-3)
                    ads_df_3_months = ads_df[ads_df['published_on'] >= pd.to_datetime(date_three_months_ago.strftime("%Y-%m-%d"), format = "%Y-%m-%d")]



                    ### Load model for prediction locally ###
                    # I did not manage to load it from Github wg_price_predictor repository using pickle, joblib nor cloudpickle
                    trained_model = get_latest_model_from_db()

            #### Analysis ####
                    placeholder = st.empty()
                    with placeholder.container():
                        st.subheader(f"""
                                This is how your WG compares to other WGs published in the past three months:
                                """)
                        col1, col2, col3, col4 = st.columns(4)

                #### Number ads ####
                        ## Same city
                        ads_df_city = ads_df_3_months[ads_df_3_months['city'] == list(ad_df_processed['city'])[0]]
                        n_posts_city = len(ads_df_city)
                        n_days_post_city = round(90/n_posts_city,1)
                        n_hours_post_city = round((24*90)/n_posts_city,1)

                        ## Same zip
                        ads_df_zip_code = ads_df_3_months[ads_df_3_months['zip_code'] == list(ad_df_processed['zip_code'])[0]]
                        n_posts_zipcode = len(ads_df_zip_code)
                        n_days_post_zipcode = round(90/n_posts_zipcode,1)
                        n_hours_post_zipcode = round((24*90)/n_posts_zipcode,1)

                        col1.markdown(f"""
                                <font size= "4">**Number of ads posted**</font>
                                """, unsafe_allow_html=True)

                        col1.markdown(f"""
                                <font size= "4">On average, <span style="color:tomato">**{n_hours_post_city if n_days_post_city <= 1 else n_days_post_city}**</span> WG ads in <span style="color:tomato">**{list(ad_df_processed['city'])[0]}**</span> were posted every {'hour' if n_days_post_city <= 1 else 'day'}.

                                <span style="color:tomato">**{n_hours_post_zipcode if n_days_post_zipcode <= 1 else n_days_post_zipcode}**</span> WG ads were posted every {'hour' if n_days_post_zipcode <= 1 else 'day'} with the same ZIP code <span style="color:tomato">**{list(ad_df_processed['zip_code'])[0]}**</span>.</font>
                                """, unsafe_allow_html=True)

                #### Size room ####
                        ## Smaller size
                        ads_df_smaller = ads_df_3_months[ads_df_3_months['size_sqm'] < list(ad_df_processed['size_sqm'])[0]]
                        percent_smaller = round(100*(len(ads_df_smaller)/len(ads_df_3_months)),1)

                        ## Smaller zip_code
                        ads_df_smaller_zipcode = ads_df_zip_code[ads_df_zip_code['size_sqm'] < list(ad_df_processed['size_sqm'])[0]]
                        percent_smaller_zipcode = round(100*(len(ads_df_smaller_zipcode)/len(ads_df_zip_code)),1)

                        col2.markdown(f"""
                                <font size= "4">**Size of the room**</font>
                                """, unsafe_allow_html=True)

                        col2.markdown(f"""
                                <font size= "4">With <span style="color:tomato">**{list(ad_df_processed['size_sqm'])[0]} sqm**</span>, this WG is bigger than <span style="color:tomato">**{percent_smaller} %**</span> of WG ads published in <span style="color:tomato">**{list(ad_df_processed['city'])[0]}**</span> and <span style="color:tomato">**{percent_smaller_zipcode} %**</span> of WG ads published with the ZIP code <span style="color:tomato">**{list(ad_df_processed['zip_code'])[0]}**</span>.</font>
                                """, unsafe_allow_html=True)

                #### Price ####
                        ## Cheaper city
                        ads_df_cheaper = ads_df_3_months[ads_df_3_months['price_euros'] < list(ad_df_processed['price_euros'])[0]]
                        percent_cheaper = round(100*(len(ads_df_cheaper)/len(ads_df_3_months)),1)

                        ## Cheaper zip_code
                        ads_df_cheaper_zipcode = ads_df_zip_code[ads_df_zip_code['price_euros'] < list(ad_df_processed['price_euros'])[0]]
                        percent_cheaper_zipcode = round(100*(len(ads_df_cheaper_zipcode)/len(ads_df_zip_code)),1)

                        col3.markdown(f"""
                                <font size= "4">**WG price**</font>
                                """, unsafe_allow_html=True)

                        col3.markdown(f"""
                                <font size= "4">Costing <span style="color:tomato">**{list(ad_df_processed['price_euros'])[0]} €**</span>, this WG is more expensive than <span style="color:tomato">**{percent_cheaper} %**</span> of WG ads published in <span style="color:tomato">**{list(ad_df_processed['city'])[0]}**</span> and <span style="color:tomato">**{percent_cheaper_zipcode} %**</span> of WG ads published with the ZIP code <span style="color:tomato">**{list(ad_df_processed['zip_code'])[0]}**</span>.</font>
                                """, unsafe_allow_html=True)

                #### Factors influencing price ####
                        # Size
                        wg_is_large = True if percent_smaller_zipcode > 50 else False

                        # Location
                        ads_df_not_zip_code = ads_df_city[ads_df_city['zip_code'] != list(ad_df_processed['zip_code'])[0]]
                        ads_df_not_zip_code = ads_df_not_zip_code[ads_df_not_zip_code['zip_code'].notna()]
                        mean_zip_code = ads_df_zip_code['price_euros'].mean()
                        mean_not_zip_code = ads_df_not_zip_code['price_euros'].mean()

                        import scipy.stats as stats
                        #perform two sample t-test with equal variances
                        p_value = stats.ttest_ind(a=ads_df_zip_code['price_euros'], b=ads_df_not_zip_code['price_euros'], equal_var = True, nan_policy = 'omit', random_state = 42)
                        if p_value[1] <=0.05:
                            if mean_zip_code > mean_not_zip_code:
                                zip_is_more = 'pricier'
                            else:
                                zip_is_more = 'cheaper'
                        else:
                            zip_is_more = 'similar'



                        # schufa_needed
                        schufa_needed = str(list(ad_df_processed['schufa_needed'])[0]) == '1'

                        # commercial_landlord
                        commercial_landlord = str(list(ad_df_processed['commercial_landlord'])[0]) == '1'

                        # capacity
                        capacity = int(list(ad_df_processed['capacity'])[0])

                        # days_available
                        days_available = int(list(ad_df_processed['days_available'])[0])

                        # wg_type_studenten
                        wg_type_studenten = str(list(ad_df_processed['wg_type_studenten'])[0]) == '1'

                        # wg_type_business
                        wg_type_business = str(list(ad_df_processed['wg_type_business'])[0]) == '1'

                        # building_type
                        building_type = str(list(ad_df_processed['building_type'])[0])


                        col4.markdown(f"""
                                <font size= "4">**Possible price factors**</font>
                                """, unsafe_allow_html=True)


                        def generate_text_possible_price_factors():
                            prompts = ['\n','- Room is large' if percent_smaller_zipcode >= 70 else '- Room is small' if percent_smaller_zipcode <= 30 else '' +\
                                '- WG in pricier neighborhood' if zip_is_more == 'pricier' else '- WG in cheaper neighborhood' if zip_is_more == 'cheaper' else '',

                                '- WGs in ' + building_type + 'building type tend to be pricier' if building_type == 'Neubau' or building_type == 'Hochhaus' else '- WGs in ' + building_type + 'building type tend to be cheaper' if building_type == 'Einfamilienhaus' else '',

                                '- Students WG type tend to be cheaper' if wg_type_studenten else '',

                                '- Business WG type tend to be pricier' if wg_type_business else '',

                                '- WGs that require Schufa tend to be pricier' if schufa_needed else '',

                                '- WGs with companies as landlord tend to be pricier' if commercial_landlord else '',

                                '- WGs with capacity for ' + str(capacity) + ' people tend to be pricier' if capacity >= 5 else '- WGs for only 2 people tend to be cheaper' if capacity == 2 else '',

                                '- Short-term rental WGs (<30 days) tend to be cheaper' if days_available <= 30 else '- WGs with open-end rental time availability tend to be cheaper' if days_available > 540 else '']

                            return '\n'.join(text for text in prompts if text != '')

                        col4.markdown(f"""
                                        <font size= "4">{generate_text_possible_price_factors()}</font>
                                        """, unsafe_allow_html=True)




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




            else:
                st.markdown("""
                            There was a problem connecting to the provided link. Please submit a valid link to a WG ad from wg-gesucht.de.

                            A valid link has the format "https://www.wg-gesucht.de/wg-zimmer-in-City-Neighborhood.1234567.html", where the Neighborhood tag may or may not be present.

                            Please get in contact if the problem persists.
                            """, unsafe_allow_html=True)



with tab2:
    st.markdown("""
            ### Have you ever wondered how your room compares to the current WG market?
            #### I've collected thousands of ads from wg-gesucht.de to answer that question for you.
            """, unsafe_allow_html=True)

    st.caption("""
        Information submitted by you in this session is not stored anywhere and won't be used for anything else other than the analysis shown here.
        The more information you give the better the analysis is, but a basic analysis can already be performed with only very little information (marked by *).
                """, unsafe_allow_html=True)

    with st.form("flat_form", clear_on_submit=True):
        placeholder = st.empty()
        with placeholder.container():
            st.subheader("""
                    \n
                    Location
                    """)
            col1, col2, col3, col4 = st.columns(4)
            col1.selectbox(label="City*", options=['<Please select>']+sorted(list(dict_city_number_wggesucht.keys())), index=0, key='city_own')
            col2.text_input("Street and house number*", value="Example Str. 15", key='address', max_chars = 100)
            col3.text_input("Neighborhood", value="", key='neighborhood', max_chars = 100)
            col4.text_input("Zip code*", value="12345", key='zip_code', max_chars = 20)

        placeholder = st.empty()
        with placeholder.container():
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


        placeholder = st.empty()
        with placeholder.container():
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

            placeholder = st.empty()
            with placeholder.container():
                col1, col2 = st.columns(2)
                col1.multiselect(label='Type of WG', options=['<Please select>','WG types'], default=None, key='wg_type')
                col2.multiselect(label='Spoken languages', options=['<Please select>','Languages'], default=None, key='languages')

        placeholder = st.empty()
        with placeholder.container():
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



        placeholder = st.empty()
        with placeholder.container():
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
        #############################
        ### Obtain main ads table ###
        #############################
        # Copying is needed to prevent subsequent steps from modifying the cached result from get_original_data()
        ads_df = get_data_from_db().copy()


        #### Checking inputted info is correct format
        if str(st.session_state.city_own) == "<Please select>":
            st.markdown("""
            Selecting a city is mandatory for analysis.
            """, unsafe_allow_html=True)
        elif str(st.session_state.address) == "Example Str. 15":
            st.markdown("""
            An address is mandatory for analysis.
            """, unsafe_allow_html=True)
        elif str(st.session_state.zip_code) == "12345":
            st.markdown("""
            The zip code is mandatory for analysis.
            """, unsafe_allow_html=True)
        # elif ~str(st.session_state.zip_code).isnumeric():
        #     st.markdown(f"""
        #     The zip code provided is invalid: {st.session_state.zip_code}.
        #     Zip code must containg only numbers.
        #     """, unsafe_allow_html=True)

        else:
            full_address = str(st.session_state.address) + ', ' + str(st.session_state.zip_code) + ', ' + str(st.session_state.city_own)

            lat, lon = geocoding_address(full_address)

            if lat != lat and lon != lon: # Check if lat and lon are nan
                st.markdown(f"""
                The provided address is incorrect: {full_address}
                """, unsafe_allow_html=True)

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
                'city': [str(st.session_state.city_own)],
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



with tab3:
    st.markdown("""
                ### This dashboard contains everything you want to know about WGs in Germany!
                To get started select below the time period, the city, and the type of market of interest and press "Show results".
                """, unsafe_allow_html=True)


    with st.form("data_overview_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        col1.selectbox("Analysis period:", ['Past week','Past month', 'Past three months', 'Past six months'],#, 'Past year'],
                       key="time_period", index=2)
        col2.selectbox("City:", ['Germany'] + sorted(list(dict_city_number_wggesucht.keys())), key="city_filter", index=0)
        col3.selectbox("Market type:", ['WG', 'Single-room flat', 'Apartment'], key="market_type", index=0)

        "---"
        submitted = st.form_submit_button("Show results")

        if submitted:
            #############################
            ### Obtain main ads table ###
            #############################
            # Copying is needed to prevent subsequent steps from modifying the cached result from get_original_data()
            ads_df = get_data_from_db().copy()


            #### Filter data for analysis
            df_filtered = filter_original_data(df = ads_df,
                                            city = st.session_state["city_filter"],
                                            time_period = st.session_state["time_period"])


            ## Filter type of offer
            market_type_df = df_filtered[df_filtered['type_offer_simple'] == st.session_state["market_type"]].reset_index().drop(columns=['index'])


            ### TODO
            ### Plot price evolution

            ### Plotting ads per market type
            placeholder = st.empty()
            with placeholder.container():
                col1, col2, col3 = st.columns([0.05,1,0.05])
                with col2:
                    st.plotly_chart(ads_per_region_stacked_barplot(df = df_filtered, time_period = st.session_state["time_period"], city = st.session_state["city_filter"]), use_container_width=True)


            ### Plotting ads per day
            placeholder = st.empty()
            with placeholder.container():
                col1, col2, col3 = st.columns([1,0.05,0.45])
                with col1:
                    st.plotly_chart(ads_per_day_stacked_barplot(df = market_type_df, city = st.session_state["city_filter"], time_period = st.session_state["time_period"],market_type = st.session_state["market_type"]), height=400, use_container_width=True)
                with col3:
                    st.plotly_chart(ads_per_hour_line_polar(df = market_type_df, city = st.session_state["city_filter"], time_period = st.session_state["time_period"],market_type = st.session_state["market_type"]), height=400, use_container_width=True)


            st.markdown(f"""
                #### Rank of rental prices in {st.session_state["city_filter"]} in the {st.session_state["time_period"].lower()} (€)
                """, unsafe_allow_html=True)

            ### Plotting ads per market type
            placeholder = st.empty()
            with placeholder.container():
                col1, col2, col3 = st.columns([0.05,1,0.05])
                with col2:
                    st.pyplot(price_rank_cities(df = df_filtered, city = st.session_state["city_filter"]))

            st.markdown("""
                *Values displayed here are **warm** rental prices that more accurately reflect living costs. Warm rent usually include the cold rent, water, heating and house maintenance costs. It may also include internet and TV/Radio/Internet taxes.
                """, unsafe_allow_html=True)

            st.markdown(f"""
                #### Square-meter prices in Germany in the {st.session_state["time_period"].lower()} (€/m²)
                """, unsafe_allow_html=True)

            placeholder = st.empty()
            with placeholder.container():
                col1, col2, col3 = st.columns([0.1,1,0.1])
                with col2:
                    st_data = st_folium(map_plotting(plotting_df=prepare_data_for_map(ads_df),market_type = st.session_state["market_type"]), width=700, height=500)

            st.markdown("""
                *Square-meter prices were calculated using the cold rent and assumes that all people living in a WG pay the same amount. This assumption is rarely true for individual WGs but works fine when several WGs are analysed together.
                **Regions without a minimum of 3 ads per ZIP code are not displayed.
                """, unsafe_allow_html=True)



with tab4:
    st.markdown(f"""
        ## Driving factors of rental prices in Germany
        """, unsafe_allow_html=True)

    st.markdown("""
                **Besides the city in which one searches for WGs, several other factors are also relevant for rental price, including the WG structure and the renting conditions.\nHere, I highlight several of these factors based on the analysis of square-meter cold rental prices (€/m²) in Germany in the past three months.**
                """, unsafe_allow_html=True)

    #############################
    ### Obtain main ads table ###
    #############################
    # Copying is needed to prevent subsequent steps from modifying the cached result from get_original_data()
    ads_df = get_data_from_db().copy()


            #### Filter data for analysis
    df_filtered = filter_original_data(df = ads_df,
                                        city = 'Germany',
                                        time_period = 'Past three months')

    placeholder = st.empty()
    with placeholder.container():
        col1, col2 = st.columns([0.5,0.4])
        with col1:
            st.markdown("""
                1.1) Business-type WGs pay higher, while student-type WGs pay lower rent.
                """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                1.2) The number of flatmates in a WG only slightly impacts rental prices.
                """, unsafe_allow_html=True)


    placeholder = st.empty()
    with placeholder.container():
        col1, col2, col3 = st.columns([0.35,0.35,0.6])
        with col1:
            df_foo = df_filtered[~df_filtered['wg_type_business'].isnull()]
            df_foo['wg_type_business'] = df_foo['wg_type_business'].map({1:'Business WGs',0:'Others'})
            st_data = st.pyplot(my_boxplot(df=df_foo,
                                            x = 'wg_type_business',
                                            x_title = "",
                                            transform_type='str',
                                            x_axis_rotation = 45,
                                            fig_height = 15))

        with col2:
            df_foo = df_filtered[~df_filtered['wg_type_studenten'].isnull()]
            df_foo['wg_type_studenten'] = df_foo['wg_type_studenten'].map({1:'Students WGs',0:'Others'})
            df_foo = df_foo.reindex(sorted(df_foo.columns, reverse = True), axis=1)
            st_data = st.pyplot(my_boxplot(df=df_foo,
                                            x = 'wg_type_studenten',
                                            x_title = "",
                                            transform_type='str',
                                            x_axis_rotation = 45,
                                            fig_height = 15,
                                            order=['Others','Students WGs']))

        with col3:
            df_foo = df_filtered.query('capacity <= 7')
            df_foo['n_flatmates'] = df_foo['capacity']-1
            st_data = st.pyplot(my_boxplot(df=df_foo,
                                        x = 'n_flatmates',
                                        x_title = "Number of flatmates",
                                        transform_type='int',
                                        font_scale=2.5))



    placeholder = st.empty()
    with placeholder.container():
        col1, col2, col3 = st.columns([0.6,0.33,0.33])
        with col1:
            st.markdown("""
            2.1) Renting a WG for less than a month is the cheapest option. Renting for a fixed long term (more than one year but less than 540 days) is more expensive than open-end WG offers.
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            2.2) WGs where the presentation of a Schufa is required for renting are generally more expensive.
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            2.3) Renting from commercial landlords (companies) strongly increases rent.
            """, unsafe_allow_html=True)


    placeholder = st.empty()
    with placeholder.container():
        col1, col2, col3 = st.columns([0.6,0.33,0.33])
        with col1:
            df_foo = df_filtered
            df_foo['rental_length_term'] = df_foo['rental_length_term'].map(
                {'30days':30,
                '90days':90,
                '180days':180,
                '270days':270,
                '365days':365,
                '540days':540,
                'plus540days':999})
            st_data = st.pyplot(my_boxplot(df=df_foo,
                                            x = 'rental_length_term',
                                            x_title = "Max rental length (days)",
                                            transform_type='int',
                                            font_scale=2.5))

        with col2:
            df_foo = df_filtered[~df_filtered['schufa_needed'].isnull()]
            df_foo['schufa_needed'] = df_foo['schufa_needed'].map({1:'Yes',0:'No'})
            st_data = st.pyplot(my_boxplot(df=df_foo,
                                            x = 'schufa_needed',
                                            x_title = "Schufa required?",
                                            transform_type='str',
                                            fig_height = 20))

        with col3:
            df_foo = df_filtered
            df_foo['commercial_landlord'] = df_foo['commercial_landlord'].map({1:'Commercial',0:'Private'})
            st_data = st.pyplot(my_boxplot(df=df_foo,
                                            x = 'commercial_landlord',
                                            x_title = "Type of landlord",
                                            transform_type='str',
                                            fig_height = 20))


    placeholder = st.empty()
    with placeholder.container():
        col1, col2, col3 = st.columns([0.1,1,0.1])
        with col2:
            st.markdown("""
            3) The type of the building strongly affects WG price. New buildings (Neubau) in particular have the most expensive offers.
            """, unsafe_allow_html=True)

            df_foo = df_filtered
            st_data = st.pyplot(my_boxplot(df=df_foo,
                                            x = 'building_type',
                                            x_title = "",
                                            transform_type='str',
                                            x_axis_rotation = 45,
                                            fig_height = 5,
                                            order='mean',
                                            font_scale=1.5))



with tab5:
    st.write('\n')
    st.markdown("""
            ### What is <span style="color:tomato">WG-prices</span>?
            WG-prices is a free and intuitive webpage where anyone can analyse the WG market in Germany. It was created by [chvieira2](https://github.com/chvieira2) out of curiosity and desire to help others. Its purpose is to help people understand the housing market better, in particular the WG market. For more details please visit [the GitHub repository](https://github.com/chvieira2/housing_crawler).

            ### Why do we need <span style="color:tomato">WG-prices</span>?
            The price paid for a WG is related to the rental price of the flat. However, the WG market is saturated, making people living in WGs susceptible to accept offers that charge more than they should. This is the case specially for younger adults and people coming from abroad that have little resources to judge the fairness of an offer.

            ### Who is <span style="color:tomato">WG-prices</span> meant for?
            WG-prices is 100% free of charge for everyone. If you live or wants to live in a WG, WG-prices helps you judge if the values charged are in accordance to the current market, or if someone is trying to exploit you. If you own a flat and is considering renting a room, WG-prices helps you decide a fair price in accordance to the current market.

            ### How does <span style="color:tomato">WG-prices</span> work?
            WG-prices collects data from wg-gesucht.de, analyses it and uses it to generate a predictive model of prices. This model is used to answer a simple question: given the current market, how much should be charged for a WG with these specifications. As the model is not perfect, the full analysis of the current market in the past 3 months in Germany is displayed in a dashboard format to help you form your own judgement.
            """, unsafe_allow_html=True)
