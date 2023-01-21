# -*- coding: utf-8 -*-

"""Housing Crawler - search for flats by crawling property portals and save them locally.

This is the dashboard/app implementation of the analysis of ads obtained from wg-gesucht.de"""

__author__ = "Carlos Henrique Vieira e Vieira"
__version__ = "1.0"
__maintainer__ = "chvieira2"
__email__ = "carloshvieira2@gmail.com"
__status__ = "Production"

from curses import BUTTON_SHIFT
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
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import statsmodels.api as sm
import scipy.stats as stats

import folium
from folium.plugins import Fullscreen
import branca.colormap
from collections import defaultdict
from folium import GeoJson, Marker
import geopandas as gpd

from housing_crawler.params import dict_city_number_wggesucht
from housing_crawler.string_utils import standardize_characters
from housing_crawler.utils import crawl_ind_ad_page2, get_data, obtain_latest_model, meters_to_coord
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
@st.cache(allow_output_mutation=True)
def get_data_from_db(file_name_tag='ads_OSM.csv', local_file_path=f'raw_data',
                     filter_ad_type = None # 'WG'
                     ):
    """
    Method to get data from local environment and return a unified dataframe

    """
    ads_db = get_data(file_name_tag=file_name_tag, local_file_path=local_file_path)

    ## Correct date format
    ads_db['published_on'] = pd.to_datetime(ads_db['published_on'], format = "%Y-%m-%d")

    ## WGs only
    if filter_ad_type is not None:
        ads_db = ads_db[ads_db['type_offer_simple'] == filter_ad_type]

    return ads_db

@st.cache
def get_latest_model_from_db():
    """
    Method to get latest trained model from local dataabse

    """
    return obtain_latest_model()

def filter_original_data(df, city, time_period, market_type_df):
    ## Format dates properly
    df['published_on'] = pd.to_datetime(df['published_on'], format = "%Y-%m-%d")

    ## Filter table
    # City of choice
    if city != 'Germany':
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





    ### Filter type of offer
    if market_type_df != 'All':
        df = df[df['type_offer_simple'] == market_type_df].reset_index().drop(columns=['index'])

    return df

def ads_per_region_stacked_barplot(df,time_period, city):

    if city == 'Germany':
        stacking_by = 'city'

        region_ads_df = df[['url', stacking_by,"type_offer_simple"]].groupby([stacking_by,"type_offer_simple"]).count().rename(columns = {'url':'count'}).sort_values(by = ['count'], ascending=False).reset_index()
    else:
        stacking_by = 'zip_code'
        df['temp'] = df['zip_code'].apply(lambda x: len(str(x)) > 3 and len(str(x)) < 6\
            and str(x) not in ['1234', '12345','0000', '1111', '2222', '3333', '4444','5555','6666','7777','8888','9999','00000', '11111', '22222', '33333', '44444','55555','66666','77777','88888','99999'] and not str(x).startswith('0') )
        df = df[df['temp']]

        ## Filter dataframe for values occuring at least 20 times in zip_code
        # Count the number of occurrences of each value in the column
        counts = df['zip_code'].value_counts()

        # Keep only the values that appear at least 20 times
        to_keep = counts[counts >= 20].index

        # Filter the dataframe
        df = df[df['zip_code'].isin(to_keep)]


        region_ads_df = df[['url', stacking_by,"type_offer_simple"]].groupby([stacking_by,"type_offer_simple"]).count().rename(columns = {'url':'count'}).sort_values(by = ['count'], ascending=False).reset_index()
        # region_ads_df = region_ads_df.head(25)


    fig = px.bar(region_ads_df, x=stacking_by, y="count", color="type_offer_simple",
            labels={
                stacking_by: "City" if stacking_by == 'city' else 'Zip code',
                "count": 'Number of similar ads',
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

def price_evolution_per_region(df,time_period, city,
                                target = 'price_per_sqm_cold'#'price_euros',
                                ):

    ## Format dates properly
    df['published_on'] = pd.to_datetime(df['published_on'], format = "%Y-%m-%d")


    ## Filter ads in between desired dates.
    date_max = pd.to_datetime(time.strftime("%Y-%m-%d", time.localtime()), format = "%Y-%m-%d")

    if time_period == 'Past week':
        date_min = datetime.date.today() + relativedelta(weeks=-1)
        date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
        agg_days = '1D'
        add_days_title = 'Daily'
    elif time_period == 'Past month':
        date_min = datetime.date.today() + relativedelta(months=-1)
        date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
        agg_days = '3D'
        add_days_title = '3-days'
    elif time_period == 'Past three months':
        date_min = datetime.date.today() + relativedelta(months=-3)
        date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
        agg_days = '5D'
        add_days_title = '5-days'
    elif time_period == 'Past six months':
        date_min = datetime.date.today() + relativedelta(months=-6)
        date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
        agg_days = '7D'
        add_days_title = 'Weekly'
    elif time_period == 'Past year':
        date_min = datetime.date.today() + relativedelta(months=-12)
        date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
        agg_days = '14D'
        add_days_title = '2-weeks'
    else:
        date_min = datetime.date.today() + relativedelta(months=-48)
        date_min = pd.to_datetime(date_min.strftime("%Y-%m-%d"), format = "%Y-%m-%d")
        agg_days = '30D'
        add_days_title = 'Monthly'

    df['temp_col'] = df['published_on'].apply(lambda x: x >= date_min and x <= date_max)
    df = df[df['temp_col']].drop(columns=['temp_col'])



    ## Filter type of offer
    if st.session_state["market_type"] != 'All':
        df = df[df['type_offer_simple'] == st.session_state["market_type"]].reset_index().drop(columns=['index'])


    ### Create the column that will be used for grouping
    start_date = pd.Timestamp('2022-08-01')
    end_date = pd.Timestamp.today()
    dates_range = pd.date_range(start_date, end_date, freq=agg_days).to_pydatetime().tolist()

    df['grouping_date'] = df['published_on'].apply(lambda x: [date for date in dates_range if date <= x][-1])



    #### Create tables to use
    # City
    df_city = df[df['city'] == city]

    # Germany
    # Add Germany average for comparison with city
    germany_ads_df = df[[target, 'grouping_date']].groupby(['grouping_date']).mean().sort_values(by = ['grouping_date'], ascending=False).reset_index()
    germany_ads_df['city'] = 'Germany'


    if city == 'Germany':
                region_ads_df = germany_ads_df
    else:
        region_ads_df = df_city[[target, 'city', 'grouping_date']].groupby(['city', 'grouping_date']).mean().sort_values(by = ['grouping_date'], ascending=False).reset_index()
        region_ads_df = pd.concat([region_ads_df, germany_ads_df])



    fig = px.line(region_ads_df, x='grouping_date', y=target, color='city',
            labels={
                'city': "Region",
                target: f'{add_days_title} average {"warm" if target == "price_euros" else "warm" if target == "price_per_sqm_warm" else "cold"} rent price ({"€" if target == "price_euros" else "€" if target == "cold_rent_euros" else "€/m²"})'
            },
            template = "ggplot2" #["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
            )
    fig.update_layout(height=300,
                        font_family="Arial",
                        margin= dict(
                        l = 10,        # left
                        r = 10,        # right
                        t = 25,        # top
                        b = 0),        # bottom
                        xaxis_title=None)

    return fig

def ads_per_day_stacked_barplot(df,city,time_period,market_type):

    if city != 'Germany':
        stacking_by = 'zip_code'

        df['temp'] = df['zip_code'].apply(lambda x: len(str(x)) > 3 and len(str(x)) < 6\
            and str(x) not in ['1234', '12345','0000', '1111', '2222', '3333', '4444','5555','6666','7777','8888','9999','00000', '11111', '22222', '33333', '44444','55555','66666','77777','88888','99999'] and not str(x).startswith('0') )
        df = df[df['temp']]

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

def predicted_vs_actual_prices_ScatterPlot(df:pd.DataFrame):

    fig= px.scatter(data_frame = df,
                    y='predicted_price_euros',x='price_euros',
                    color = 'type_offer_simple',
            labels={
                "predicted_price_euros": "Predicted rental price (€)",
                "price_euros": "Actual rental price (€)",
                "type_offer_simple": "Market asset"
            },
            template = "seaborn"
            )
    fig.update_layout(showlegend=True if len(set(df['type_offer_simple'])) > 1 else False,
                      height=300,
                        font_family="Arial",
                        margin= dict(
                        l = 10,        # left
                        r = 10,        # right
                        t = 25,        # top
                        b = 0        # bottom
                        )
                        )

    return fig

def prediction_error_boxplot(df:pd.DataFrame):
    if len(set(df['city'])) == 1:
        stacking_by = 'zip_code'

        df = df[df['zip_code'].notnull()]
        df['zip_code'] = df['zip_code'].astype(int).astype(str)
        df.sort_values('zip_code', inplace=True)

        ## Filter dataframe for values occuring at least 3 times in zip_code and type_offer_simple
        df['zip_code_type_offer_simple'] = df['zip_code'].astype(str) + df['type_offer_simple'].astype(str)
        for col in ['zip_code','zip_code_type_offer_simple']:
            # Count the number of occurrences of each value in the column
            counts = df[col].value_counts()

            # Keep only the values that appear at least 3 times
            to_keep = counts[counts >= 3].index

            # Filter the dataframe
            df = df[df[col].isin(to_keep)]

    else:
        stacking_by = 'city'

    fig= px.box(df, x=stacking_by, y="residuals", color='type_offer_simple',
                labels={
                    stacking_by: '' if stacking_by == 'city' else 'Zip code',
                    "residuals": 'Prediction error (€)',
                    "type_offer_simple": "Market asset"
                },
                template = "seaborn",
                points=False)

    if stacking_by == 'city':
        fig.update_xaxes(categoryorder='array', categoryarray= sorted(list(dict_city_number_wggesucht.keys())))

    fig.update_layout(yaxis_range=[-500,500],
                    showlegend=True if len(set(df['type_offer_simple'])) > 1 else False,
                      height=300,
                        font_family="Arial",
                        margin= dict(
                        l = 10,        # left
                        r = 10,        # right
                        t = 25,        # top
                        b = 0,        # bottom
                ))

    return fig

def fraction_prediction_error_barplot(df:pd.DataFrame, market_type: str):
    if len(set(df['city'])) == 1:
        stacking_by = 'zip_code'

        df = df[df['zip_code'].notnull()]
        df['zip_code'] = df['zip_code'].astype(int).astype(str)
        df.sort_values('zip_code', inplace=True)

        ## Filter dataframe for values occuring at least 3 times in zip_code
        # Count the number of occurrences of each value in the column
        counts = df['zip_code'].value_counts()

        # Keep only the values that appear at least 3 times
        to_keep = counts[counts >= 3].index

        # Filter the dataframe
        df = df[df['zip_code'].isin(to_keep)]

    else:
        stacking_by = 'city'


    ### Create table for plotting
    df['residuals_abs'] = abs(df['residuals'])

    count_all = df[['id', stacking_by,'type_offer_simple']].groupby([stacking_by,'type_offer_simple']).count().rename(columns={'id':'count_all'})
    count_all['count_0'] = round(100*count_all['count_all']/count_all['count_all'],1)

    for filter in [25,50,100,200,400]:
        _foo = df[df['residuals_abs'] <=filter]
        count_foo = _foo[['id', stacking_by,'type_offer_simple']].groupby([stacking_by,'type_offer_simple']).count().rename(columns={'id':f'count_{filter}'})
        count_foo[f'count_{filter}'] = round(100*count_foo[f'count_{filter}']/count_all['count_all'],1)
        count_all = pd. merge(count_all,count_foo, left_index=True, right_index=True)
    count_all = count_all.reset_index()

    ## Subtract columns for plotting
    count_all['count_400'] = count_all['count_400']-count_all['count_200']
    count_all['count_200'] = count_all['count_200']-count_all['count_100']
    count_all['count_100'] = count_all['count_100']-count_all['count_50']
    # count_all['count_50'] = count_all['count_50']-count_all['count_25']


    ## Plotting
    _filtered = count_all[count_all['type_offer_simple'] == market_type]
    x = _filtered[stacking_by]
    fig = go.Figure()
    # fig.add_trace(go.Bar(x=x, y=_filtered['count_25'], name='25'))
    fig.add_trace(go.Bar(x=x, y=_filtered['count_50'], name='50'))
    fig.add_trace(go.Bar(x=x, y=_filtered['count_100'], name='100'))
    fig.add_trace(go.Bar(x=x, y=_filtered['count_200'], name='200'))
    fig.add_trace(go.Bar(x=x, y=_filtered['count_400'], name='400'))

    fig.update_layout(height=250,
                      barmode='stack',
                      template = "ggplot2",
                      legend_title="Error threshold",
                        yaxis_title="Fraction of ads (%)",
                        font_family="Arial",
                        margin= dict(
                        l = 10,        # left
                        r = 10,        # right
                        t = 0,        # top
                        b = 0,        # bottom
                ))

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

        df['temp'] = df['zip_code'].apply(lambda x: len(str(x)) > 3 and len(str(x)) < 6\
            and str(x) not in ['1234', '12345','0000', '1111', '2222', '3333', '4444','5555','6666','7777','8888','9999','00000', '11111', '22222', '33333', '44444','55555','66666','77777','88888','99999'] and not str(x).startswith('0') )
        df = df[df['temp']]

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
    city_wg_df = city_wg_df[city_wg_df['count']>= 20]
    city_singleroom_df = city_singleroom_df[city_singleroom_df['count']>= 20]
    city_flathouse_df = city_flathouse_df[city_flathouse_df['count']>= 20]

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

def selectbox_to_simplified_german(feature : str):

    if feature == '<Please select>':
        return np.nan
    else:
        try:
            feature_list = feature.split('/')[1]
            return feature_list if len(feature_list) > 0 else np.nan
        except IndexError:
            feature_list =  feature[0]
            return feature_list if feature_list != '' else np.nan

def url_to_df(url_for_search):
    """
    Receives a URL, analyses it with crawl_ind_ad_page2 and processes it with process_ads_tables.
    Returns the dataframe ready for prediction with the model.
    """
    df = crawl_ind_ad_page2(url_for_search)
    ## Process ad_df for analysis
    return process_ads_tables(input_ads_df = df, save_processed = False, df_feats_tag = 'city')

def pred_from_df(ad_df):
    """
    This function receives a processed dataframe from url_to_df() and returns the predicted cold rent price.
    """
    ### Load model for prediction locally ###
    # I did not manage to load it from Github wg_price_predictor repository using pickle, joblib nor cloudpickle
    trained_model = get_latest_model_from_db()


    # Predict expected cold_rent_euros
    pred_price_warm = float(trained_model.predict(ad_df))
    return int(float(pred_price_warm))

@st.cache(allow_output_mutation=True)
def analyse_df_ad(ads_db: pd.DataFrame, ad_df: pd.DataFrame):
    """
    This function receives a dataframe produced by url_to_df and the database of all ads for analysis.
    Function returns a dictionary analysis_dict containing all analysis values
    """
    analysis_dict={}

    ## Remove ad of interest from database
    ads_db = ads_db[ads_db['id'] != list(ad_df['id'])[0]]

    ## Filter for past 3 months only
    date_three_months_ago = datetime.date.today() + relativedelta(months=-3)
    ads_df_3_months = ads_db[ads_db['published_on'] >= pd.to_datetime(date_three_months_ago.strftime("%Y-%m-%d"), format = "%Y-%m-%d")]



    #### Number ads ####
    ## Same city
    ads_df_city = ads_df_3_months[ads_df_3_months['city'] == list(ad_df_processed['city'])[0]]
    n_posts_city = len(ads_df_city)
    analysis_dict['n_days_post_city'] = round(90/n_posts_city,1)
    analysis_dict['n_hours_post_city'] = round((24*90)/n_posts_city,1)

    ## Same zip
    ads_df_zip_code = ads_df_3_months[ads_df_3_months['zip_code'] == list(ad_df_processed['zip_code'])[0]]
    n_posts_zipcode = len(ads_df_zip_code)
    analysis_dict['n_days_post_zipcode'] = round(90/n_posts_zipcode,1)
    analysis_dict['n_hours_post_zipcode'] = round((24*90)/n_posts_zipcode,1)



    #### Size room ####
    ## Smaller size
    ads_df_smaller = ads_df_3_months[ads_df_3_months['size_sqm'] < list(ad_df_processed['size_sqm'])[0]]
    analysis_dict['percent_smaller'] = round(100*(len(ads_df_smaller)/len(ads_df_3_months)),1)

    ## Smaller zip_code
    ads_df_smaller_zipcode = ads_df_zip_code[ads_df_zip_code['size_sqm'] < list(ad_df_processed['size_sqm'])[0]]
    analysis_dict['percent_smaller_zipcode'] = round(100*(len(ads_df_smaller_zipcode)/len(ads_df_zip_code)),1)



    #### Price ####
    ## Cheaper city
    ads_df_cheaper = ads_df_3_months[ads_df_3_months['price_euros'] < list(ad_df_processed['price_euros'])[0]]
    analysis_dict['percent_cheaper'] = round(100*(len(ads_df_cheaper)/len(ads_df_3_months)),1)

    ## Cheaper zip_code
    ads_df_cheaper_zipcode = ads_df_zip_code[ads_df_zip_code['price_euros'] < list(ad_df_processed['price_euros'])[0]]
    analysis_dict['percent_cheaper_zipcode'] = round(100*(len(ads_df_cheaper_zipcode)/len(ads_df_zip_code)),1)



    #### Factors influencing price ####
    # Size
    analysis_dict['wg_is_large'] = True if analysis_dict['percent_smaller_zipcode'] > 50 else False

    # Location
    ads_df_not_zip_code = ads_df_city[ads_df_city['zip_code'] != list(ad_df_processed['zip_code'])[0]]
    ads_df_not_zip_code = ads_df_not_zip_code[ads_df_not_zip_code['zip_code'].notna()]
    mean_zip_code = ads_df_zip_code['price_euros'].mean()
    mean_not_zip_code = ads_df_not_zip_code['price_euros'].mean()

    #perform two sample t-test with equal variances
    p_value = stats.ttest_ind(a=ads_df_zip_code['price_euros'], b=ads_df_not_zip_code['price_euros'], equal_var = True, nan_policy = 'omit', random_state = 42)
    if p_value[1] <=0.05:
        if mean_zip_code > mean_not_zip_code:
            analysis_dict['zip_is_more'] = 'pricier'
        else:
            analysis_dict['zip_is_more'] = 'cheaper'
    else:
        analysis_dict['zip_is_more'] = 'similar'



    # schufa_needed
    analysis_dict['schufa_needed'] = str(list(ad_df_processed['schufa_needed'])[0]) == '1'

    # commercial_landlord
    analysis_dict['commercial_landlord'] = str(list(ad_df_processed['commercial_landlord'])[0]) == '1'

    # capacity
    analysis_dict['capacity'] = int(list(ad_df_processed['capacity'])[0])

    # days_available
    analysis_dict['days_available'] = int(list(ad_df_processed['days_available'])[0])

    # wg_type_studenten
    analysis_dict['wg_type_studenten'] = str(list(ad_df_processed['wg_type_studenten'])[0]) == '1'

    # wg_type_business
    analysis_dict['wg_type_business'] = str(list(ad_df_processed['wg_type_business'])[0]) == '1'

    # building_type
    analysis_dict['building_type'] = str(list(ad_df_processed['building_type'])[0])




    #### Price prediction ####
    try:
        cold_rent = int(list(ad_df_processed['cold_rent_euros'])[0])

    except ValueError: # cold_rent_euros is nan
        # create predictive model for cold rent from warm rent
        wg_df_foo = ads_df_3_months[ads_df_3_months['cold_rent_euros'].notna()]
        wg_df_foo = wg_df_foo[wg_df_foo['price_euros'].notna()]
        model_cold_from_warm = sm.OLS(wg_df_foo.cold_rent_euros, wg_df_foo.price_euros).fit()

        # # Add cold rent predictions only if cold_rent_euros is nan
        ad_df_processed['cold_rent_euros'] = int(model_cold_from_warm.predict(ad_df_processed['price_euros'])[0])

        cold_rent = int(list(ad_df_processed['cold_rent_euros'])[0])


    # Predict expected cold_rent_euros
    try:
        #### Currently the prediction isn't working with XGBoost and this will always fail
        analysis_dict['cold_rent_pred'] = pred_from_df(ad_df = ad_df_processed)

        # Calculate extra costs
        extra_costs_total = int(list(ad_df_processed['price_euros'])[0]) - int(cold_rent)

        analysis_dict['warm_rent_pred'] =  int(analysis_dict['cold_rent_pred'] + extra_costs_total)


        ## Ad evaluation
        analysis_dict['ad_evaluation'] = 'High price' if int(ad_df_processed['price_euros']) > analysis_dict['warm_rent_pred']*1.2 else 'Low price' if int(ad_df_processed['price_euros']) < analysis_dict['warm_rent_pred']*1.2 else 'Fairly priced'
    except:
        analysis_dict['ad_evaluation'] = None

    return analysis_dict


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
                                        'Predicting WG prices with AI', 'About'])

with tab1:
    st.markdown("""
            ### Check how <span style="color:tomato">**a WG room ad**</span> compares to thousands of other ads posted on [WG-gesucht](https://www.wg-gesucht.de/).
            """, unsafe_allow_html=True)

    with st.form("entry_url", clear_on_submit=False):
        st.text_input("Your wg-gesucht link:", value="", key='url', max_chars = 250)

        submitted_url = st.form_submit_button("Submit")




    if submitted_url:


        #### Preparing for analysis ####
        url = str(st.session_state["url"]).strip()
        url_ok = False
        if url == '' or url == 'https://www.wg-gesucht.de/wg-zimmer-in-City-Neighborhood.1234567.html' or not url.startswith('https://www.wg-gesucht.de'):
            st.markdown("""
                Please submit a valid link to a WG ad from wg-gesucht.de.
                """, unsafe_allow_html=True)

        elif not url.startswith('https://www.wg-gesucht.de/wg-zimmer-'):
            st.markdown("""
                The ad is not a WG. At the moment only WGs are supported.
                """, unsafe_allow_html=True)

        else:
            url_ok = True


        if url_ok:
            with st.spinner(f'Analysing {url}. This could take a minute.'):
                ## Process url to obtain table for prediction
                ad_df_processed = None
                try:
                    ad_df_processed = url_to_df(url_for_search = url)
                    st.success('Analysis was successful!', icon="✅")

                except Exception as err:
                    st.error(f"""
                                The analysis failed. Most common reasons for analysis to fail are:

                                - The ad is not active
                                - Posted rent price is unrealistically expensive/cheap
                                - Posted room size is unrealistically large/small
                                - Invalid entries in other fields in the ad page

                                Please get in contact if you think none of these reasons apply in your case.
                                """, icon="🚨")
                    print(f"Unexpected {err=}, {type(err)=}")
                    # raise



            if ad_df_processed is not None:

        #### Collect files needed for analysis ####
                ### Obtain main ads table ###
                # Copying is needed to prevent subsequent steps from modifying the cached result from get_original_data()
                ads_df = get_data_from_db(filter_ad_type = 'WG').copy()

                ### Filter data for analysis ###
                ad_df_analysis_dict = analyse_df_ad(ads_db = ads_df, ad_df = ad_df_processed)


        #### Analysis results ####
                with st.container():
                    st.subheader(f"""
                            This is how your WG compares to other WGs published in the past three months:
                            """)
                    col1, col2, col3, col4 = st.columns(4)

        #### Number ads ####
                    col1.markdown(f"""
                            <font size= "4">**Number of ads posted**</font>
                            """, unsafe_allow_html=True)

                    col1.markdown(f"""
                            <font size= "4">On average, <span style="color:tomato">**{ad_df_analysis_dict['n_hours_post_city'] if ad_df_analysis_dict['n_days_post_city'] <= 1 else ad_df_analysis_dict['n_days_post_city']}**</span> WG room ads were posted in <span style="color:tomato">**{list(ad_df_processed['city'])[0]}**</span> every {'hour' if ad_df_analysis_dict['n_days_post_city'] <= 1 else 'day'}.

                            <span style="color:tomato">**{ad_df_analysis_dict['n_hours_post_zipcode'] if ad_df_analysis_dict['n_days_post_zipcode'] <= 1 else ad_df_analysis_dict['n_days_post_zipcode']}**</span> WG room ads with the same ZIP code <span style="color:tomato">**{list(ad_df_processed['zip_code'])[0]}**</span> were posted every {'hour' if ad_df_analysis_dict['n_days_post_zipcode'] <= 1 else 'day'}.</font>
                            """, unsafe_allow_html=True)

        #### Size room ####
                    col2.markdown(f"""
                            <font size= "4">**Size of the room**</font>
                            """, unsafe_allow_html=True)

                    col2.markdown(f"""
                            <font size= "4">With <span style="color:tomato">**{list(ad_df_processed['size_sqm'])[0]} sqm**</span>, this WG room is bigger than <span style="color:tomato">**{ad_df_analysis_dict['percent_smaller']} %**</span> of WG rooms in <span style="color:tomato">**{list(ad_df_processed['city'])[0]}**</span> and <span style="color:tomato">**{ad_df_analysis_dict['percent_smaller_zipcode']} %**</span> of WG rooms with the same ZIP code <span style="color:tomato">**{list(ad_df_processed['zip_code'])[0]}**</span>.</font>
                            """, unsafe_allow_html=True)

        #### Price ####
                    col3.markdown(f"""
                            <font size= "4">**WG price**</font>
                            """, unsafe_allow_html=True)

                    col3.markdown(f"""
                            <font size= "4">Costing <span style="color:tomato">**{list(ad_df_processed['price_euros'])[0]} €**</span>, this WG room is more expensive than <span style="color:tomato">**{ad_df_analysis_dict['percent_cheaper']} %**</span> of WG rooms in <span style="color:tomato">**{list(ad_df_processed['city'])[0]}**</span> and <span style="color:tomato">**{ad_df_analysis_dict['percent_cheaper_zipcode']} %**</span> of WG rooms with the same ZIP code <span style="color:tomato">**{list(ad_df_processed['zip_code'])[0]}**</span>.</font>
                            """, unsafe_allow_html=True)

        #### Factors influencing price ####
                    col4.markdown(f"""
                            <font size= "4">**Possible factors affecting price**</font>
                            """, unsafe_allow_html=True)


                    def generate_text_possible_price_factors():
                        prompts = ['\n','- Room is large' if ad_df_analysis_dict['percent_smaller_zipcode'] >= 70 else '- Room is small' if ad_df_analysis_dict['percent_smaller_zipcode'] <= 30 else '' +\
                            '- WG in pricier neighborhood' if ad_df_analysis_dict['zip_is_more'] == 'pricier' else '- WG in cheaper neighborhood' if ad_df_analysis_dict['zip_is_more'] == 'cheaper' else '',

                            '- WGs in ' + ad_df_analysis_dict['building_type'] + ' building type tend to be pricier' if ad_df_analysis_dict['building_type'] == 'Neubau' or ad_df_analysis_dict['building_type'] == 'Hochhaus' else '- WGs in ' + ad_df_analysis_dict['building_type'] + ' building type tend to be cheaper' if ad_df_analysis_dict['building_type'] == 'Einfamilienhaus' else '',

                            '- Students WG type tend to be cheaper' if ad_df_analysis_dict['wg_type_studenten'] else '',

                            '- Business WG type tend to be pricier' if ad_df_analysis_dict['wg_type_business'] else '',

                            '- WGs that require Schufa tend to be pricier' if ad_df_analysis_dict['schufa_needed'] else '',

                            '- WGs with companies as landlord tend to be pricier' if ad_df_analysis_dict['commercial_landlord'] else '',

                            '- WGs with capacity for ' + str(ad_df_analysis_dict['capacity']) + ' people tend to be pricier' if ad_df_analysis_dict['capacity'] >= 5 else '- WGs for only 2 people tend to be cheaper' if ad_df_analysis_dict['capacity'] == 2 else '',

                            '- Short-term rental WGs (<30 days) tend to be cheaper' if ad_df_analysis_dict['days_available'] <= 30 else '- WGs with open-end rental time availability tend to be cheaper' if ad_df_analysis_dict['days_available'] > 540 else '']

                        return '\n'.join(text for text in prompts if text != '')

                    col4.markdown(f"""
                                    <font size= "4">{generate_text_possible_price_factors()}</font>
                                    """, unsafe_allow_html=True)


        #### Price prediction ####
                if ad_df_analysis_dict['ad_evaluation'] is not None:
                    '\n'
                    '\n'
                    st.markdown(f"""
                            <font size= "6">**Rent price fairness**</font>
                            """, unsafe_allow_html=True)
                    with st.container():

                        if ad_df_analysis_dict['ad_evaluation'] == 'High price':
                            ad_evaluation = f"""After taking the margin of error in consideration, we consider this warm rent price to be significantly above the price predicted by AI our model. Therefore the offered price has a <span style="color:tomato">**HIGH**</span> price in our opinion"""
                        elif ad_df_analysis_dict['ad_evaluation'] == 'Fairly priced':
                            ad_evaluation = f"""After taking the margin of error in consideration, we consider this warm rent price to be in accordance with our model prediction. Therefore the offered price is in our opinion <span style="color:tomato">**FAIRLY PRICED**</span>"""
                        elif ad_df_analysis_dict['ad_evaluation'] == 'Low price':
                            ad_evaluation = f"""After taking the margin of error in consideration, we consider this warm rent price to be significantly below the price predicted by AI our model. Therefore the offered price has a <span style="color:tomato">**LOW**</span> price in our opinion"""

                        # Display predictions
                        st.markdown(f"""
                                    The predicted warm rent for this offer is: <span style="color:tomato">**{ad_df_analysis_dict['warm_rent_pred']} €**</span>. This prediction is composed of the predicted cold rent ({ad_df_analysis_dict['cold_rent_pred']} €), plus invariant mandatory and extra costs taken from the ad page. In the ad page, this WG room is priced at <span style="color:tomato">**{int(ad_df_processed['price_euros'])} €**</span> (warm rent). {ad_evaluation}.
                                    """, unsafe_allow_html=True)

                        st.markdown(f"""
                                <font size= "2">*There are two types of rent prices: cold and warm rent. Cold rent is the basic price of the rent, while warm rent usually includes the cold rent, water, heating and house maintenance costs. Warm rent may also include internet and TV/radio/internet taxes and other invariant costs.</font>
                                """, unsafe_allow_html=True)


        #### WG room recommendations ####
                with st.container():
                    '\n'
                    '\n'
                    st.markdown(f"""
                            <font size= "6">**Similar WG rooms**</font>
                            """, unsafe_allow_html=True)
                    with st.spinner(f'Obtaining similar WG rooms.'):

                        ## Filter city
                        ads_df_recommendation = ads_df[ads_df['city'] == list(ad_df_processed['city'])[0]]

                        ## Filter distance
                        lat_ad = float(list(ad_df_processed['latitude'])[0])
                        lon_ad = float(list(ad_df_processed['longitude'])[0])

                        for _day in [5,4,3,2,1]:
                            for _price in [0.3,0.2,0.1,0.05]:
                                for _square in [12000,8000,4000,2000,1000,500]:
                                    if len(ads_df_recommendation) > 10 or len(ads_df_recommendation)<3:

                                        lat_max = lat_ad + meters_to_coord(_square, latitude=lat_ad, direction='north')
                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['latitude'] <=lat_max]

                                        lat_min = lat_ad - meters_to_coord(_square, latitude=lat_ad, direction='south')
                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['latitude'] >=lat_min]

                                        lon_max = lon_ad + meters_to_coord(_square, latitude=lat_ad, direction='west')
                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['longitude'] <=lon_max]

                                        lon_min = lon_ad - meters_to_coord(_square, latitude=lat_ad, direction='east')
                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['longitude'] >=lon_min]


                                        ## Filter price
                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['price_euros'] >= int(list(ad_df_processed['price_euros'])[0])*(1-_price)]
                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['price_euros'] <= int(list(ad_df_processed['price_euros'])[0])*(1+_price)]


                                        ## Filter publication date
                                        selected_days_ago = datetime.date.today() + relativedelta(days=-_day)

                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['published_on'] >= pd.to_datetime(selected_days_ago.strftime("%Y-%m-%d"), format = "%Y-%m-%d")]
                                    else:
                                        break



                    if len(ads_df_recommendation) > 0:
                        evaluations = []
                        for _foo in range(1,len(ads_df_recommendation)+1):
                            _df = ads_df_recommendation.head(_foo).tail(1)
                            evaluations.append(analyse_df_ad(ads_db = ads_df, ad_df = _df)['ad_evaluation'])

                        recomm_ads = {}
                        recomm_ads['Ad title'] = list(ads_df_recommendation['title'])
                        recomm_ads['WG address'] = list(ads_df_recommendation['address'])
                        recomm_ads['Price (€)'] = list(ads_df_recommendation['price_euros'].astype('int'))
                        recomm_ads['Room size (sqm)'] = list(ads_df_recommendation['size_sqm'].astype('int'))
                        recomm_ads['Flatmates'] = list(ads_df_recommendation['capacity'].astype('int')-1)
                        recomm_ads['Available from'] = list(ads_df_recommendation['available_from'])
                        recomm_ads['Available until'] = [item if item == item else "Open end" for item in list(ads_df_recommendation['available_to'])]
                        recomm_ads['Our price evaluation'] = evaluations
                        recomm_ads['Link'] = list(ads_df_recommendation['url'])

                        # Create dataframe
                        recomm_ads = pd.DataFrame.from_dict(recomm_ads)


                        recomm_ads
                    else:
                        st.markdown(f"""
                                <font size= "4">We could not find similar WG rooms recently published on wg-gesucht.de.</font>
                                """, unsafe_allow_html=True)





        else:
            st.markdown("""
                        There was a problem connecting to the provided link. Please submit a valid link to a WG ad from wg-gesucht.de.

                        A valid link has the format "https://www.wg-gesucht.de/wg-zimmer-in-City-Neighborhood.1234567.html", where the Neighborhood tag may or may not be present.

                        Please get in contact if the problem persists.
                        """, unsafe_allow_html=True)



with tab2:
    st.markdown("""
            ### Check how <span style="color:tomato">**your own WG room**</span> compares to thousands of other WG rooms ads posted on [WG-gesucht](https://www.wg-gesucht.de/).
            """, unsafe_allow_html=True)

    st.caption("""
        Information submitted by you in this session is not stored anywhere and won't be used for anything else other than the analysis shown here.
        The more information you give the better the analysis is, but a basic analysis can already be performed with only very little information (marked by *).
                """, unsafe_allow_html=True)

    with st.expander('WG room info', expanded=True):
        with st.form("flat_form", clear_on_submit=False):
            with st.container():
                st.subheader("""
                        \n
                        Location
                        """)
                col1, col2, col3 = st.columns(3)
                _city = col1.selectbox(label="City/Ort*", options=['<Please select>']+sorted(list(dict_city_number_wggesucht.keys())), index=0)

                _street_number = col2.text_input("Street and house number/Straße und Hausnummer*", value="Str 15", max_chars = 100)

                # col3.text_input("Neighborhood/Stadtteil", value="", key='neighborhood', max_chars = 100)

                _zip_code = col3.text_input("Zip code/Postleitzahl*", value="12345", max_chars = 20)


            with st.container():
                st.subheader("""
                        \n
                        Price and costs
                        """)
                col1, col2, col3 = st.columns(3)
                _price_euros = col1.number_input(label='Total rent/Gesamtmiete (€)*', value=0, step=1)

                _cold_rent_euros = col2.number_input(label='Cold rent/Kaltmiete (€)', value=0, step=1)

                _mandatory_costs_euros = col3.number_input(label='Mandatory costs/Nebenkosten (€)', value=0, step=1)

                _extra_costs_euros = col1.number_input(label='Extra costs/Sonstige Kosten (€)', value=0, step=1)

                _transfer_costs_euros = col2.number_input(label='Transfer costs/Ablösevereinbarung (€)', value=0, step=1)

                _deposit = col3.number_input(label='Deposit/Kaution (€)', value=0, step=1)


            with st.container():
                st.subheader("""
                        \n
                        Information about the room and building
                        """)
                col1, col2, col3, col4 = st.columns(4)
                _size_sqm = col1.number_input(label='Room size/Zimmergröße (m²)*', min_value=0, max_value=60, value=0, step=1)

                _home_total_size = col2.number_input(label='Flat size/Wohnungsgröße (m²)', min_value=0, max_value=250, value=0, step=1)

                _building_type = col3.selectbox(label="Type of building/Haustyp", options=['<Please select>','Altbau', 'sanierter Altbau', 'Neubau', 'Reihenhaus', 'Doppelhaus', 'Einfamilienhaus', 'Mehrfamilienhaus', 'Hochhaus', 'Plattenbau'], index=0)
                _building_type = '' if _building_type == '<Please select>' else _building_type


                _floor = col4.selectbox(label="Floor/Etage", options=['<Please select>','Basement/Keller', 'Low ground floor/Tiefparterre','Ground floor/EG','High ground floor/Hochparterre','1st floor/1. OG','2nd floor/2. OG','3rd floor/3. OG','4th floor/4. OG','5th floor/5. OG','6th floor or higher/höher als 5. OG','Attic/Dachgeschoss'], index=0)
                _floor = selectbox_to_simplified_german(_floor)



                _parking = col1.selectbox(label='Parking condition/Parksituation', options=['<Please select>','Good parking facilities/gute Parkmöglichkeiten', 'Bad parking facilities/schlechte Parkmöglichkeiten', 'Resident parking/Bewohnerpark', 'Own parking/eigener Parkplatz', 'Underground garage/Tiefgaragenstellplatz'], index=0)
                _parking = selectbox_to_simplified_german(_parking)

                _distance_public_transport = col2.select_slider(label='Walking distance to public transport/ÖPNV (minutes)', options=[str(n) for n in range(0,61)], value='0')
                _distance_public_transport = str(_distance_public_transport) + ' Minuten zu Fuß entfernt'

                _barrier_free = col3.selectbox("Barrier-free/Barrierefrei ", ['<Please select>','Suitable for wheelchair/geeignet für Rollstuhlfaher','Not suitable for wheelchair/ungeeignet für Rollstuhlfaher'], index=0)
                _barrier_free = selectbox_to_simplified_german(_barrier_free)

                _schufa_needed = col4.selectbox("Schufa requested/erwünscht", ['<Please select>','Yes', 'No'], index=0)


            with st.container():
                st.subheader("""
                        \n
                        WG-info
                        """)
                col1, col2, col3 = st.columns(3)
                _female_flatmates = col1.select_slider(label='Female flatmates/weibliche Mitbewohnerinnen', options=[str(n) for n in range(0,11)], value='0')

                _male_flatmates = col2.select_slider(label='Male flatmates/männliche Mitbewohner', options=[str(n) for n in range(0,11)], value='0')

                _diverse_flatmates = col3.select_slider(label='Diverse flatmates/diverse Mitbewohner*innen', options=[str(n) for n in range(0,11)], value='0')


                _gender_searched_flatmate = col1.selectbox("Gender searched/gesuchtes Geschlecht", ['Gender not relevant/Geschlecht egal','Woman/Frau', 'Man/Mann', 'Diverse/Divers'], index=0)
                _gender_searched_flatmate = selectbox_to_simplified_german(_gender_searched_flatmate)

                _minAge_searched_flatmate = col2.select_slider(label='Minimal searched age/Minimales gesuchtes Alter', options=[str(n) for n in range(0,100)], value='0')

                _maxAge_searched_flatmate = col3.select_slider(label='Maximal searched age/maximales gesuchtes Alter', options=[str(n) for n in range(0,100)], value='0')

                if _minAge_searched_flatmate == '0' and _maxAge_searched_flatmate == '0':
                    _gender_search = _gender_searched_flatmate
                elif _minAge_searched_flatmate == '0' and _maxAge_searched_flatmate != '0':
                    _gender_search = _gender_searched_flatmate + ' bis ' + str(_maxAge_searched_flatmate) + ' Jahren'
                elif _minAge_searched_flatmate != '0' and _maxAge_searched_flatmate == '0':
                    _gender_search = _gender_searched_flatmate + ' ab ' + str(_maxAge_searched_flatmate) + ' Jahren'
                elif _minAge_searched_flatmate != '0' and _maxAge_searched_flatmate != '0':
                    _gender_search = _gender_searched_flatmate + ' zwischen ' + str(_minAge_searched_flatmate) + ' und '+ str(_maxAge_searched_flatmate) + ' Jahren'



                _min_age_flatmates = col1.select_slider(label='Mininum age of flatmates/Mindestalter', options=[str(n) for n in range(0,100)], value='0')

                _max_age_flatmates = col2.select_slider(label='Maximal age of flatmates/Höchstalter', options=[str(n) for n in range(0,100)], value='0')

                if _min_age_flatmates == '0' and _max_age_flatmates == '0':
                    _age_range = ''
                elif _min_age_flatmates == '0' and _max_age_flatmates != '0':
                    _age_range = 'bis ' + str(_max_age_flatmates) + ' Jahre'
                elif _min_age_flatmates != '0' and _max_age_flatmates == '0':
                    _age_range = 'ab ' + str(_max_age_flatmates) + ' Jahre'
                elif _min_age_flatmates != '0' and _max_age_flatmates != '0':
                    _age_range = str(_min_age_flatmates) + ' bis '+ str(_max_age_flatmates) + ' Jahre'


                _smoking = col3.selectbox("Smoking/Rauchen", ['<Please select>','Allowed everywhere/Rauchen überall erlaubt', 'Allowed in your room/Rauchen im Zimmer erlaubt', 'Allowed on the balcony/Rauchen auf dem Balkon erlaubt', 'No smoking/Rauchen nicht erwünscht'], index=0)
                _smoking = selectbox_to_simplified_german(_smoking)


                with st.container():
                    col1, col2 = st.columns(2)
                    _wg_type = col1.multiselect(label='WG types/WG-Arten', options=['Studenten-WG','Berufstätigen-WG','Zweck-WG','keine Zweck-WG','gemischte WG','Frauen-WG','Männer-WG','WG mit Kindern','Plus-WG','Business-WG','Verbindung','Mehrgenerationen','Wohnheim','LGBTQIA+','Azubi-WG','Vegetarisch/Vegan','Senioren-WG','Wohnen für Hilfe','Alleinerziehende','inklusive WG','Internationals welcome','funktionale WG','WG-Neugründung'], default=None)
                    _wg_type = [item.split('/')[1] for item in _wg_type]

                    _languages = col2.multiselect(label='Languages/Sprachen', options=['German/Deutsch', 'English/Englisch','Spanish/Spanisch', 'Italian/Italienisch', 'French/Französisch', 'Turkish/Türkisch', 'Albanian/Albanisch', 'Arabic/Arabisch','Bengali', 'Bosnian/Bosnisch','Chinese/Chinesisch', 'Finish/Finnisch','Greek/Griechisch', 'Hindi','Danish/Dänisch', 'Japansese/Japanisch','Croatian/Kroatisch', 'Dutch/Niederländisch','Norwegian/Norwegisch', 'Polish/Polnisch','Portuguese/Portugiesisch', 'Romenian/Rumänisch','Russian/Russisch','Swedish/Schwedisch', 'Serbian/Serbisch','Slovenian/Slowenisch','Czech/Tschechisch', 'Hungarian/Ungarisch','Sign language/Gebärdensprache'], default=None)
                    _languages = ", ".join([item.split('/')[1] for item in _languages])


            with st.container():
                st.subheader("""
                        \n
                        Energy and power
                        """)
                col1, col2, col3 = st.columns(3)
                _energy_certificate = col1.selectbox(label='Certification type/Energieausweistyp', options=['<Please select>','Requirement/Bedarfausweis','Consumption/Verbrauchausweis'], index=0)
                _energy_certificate = selectbox_to_simplified_german(_energy_certificate)

                _heating_energy_source = col1.selectbox(label='Heating energy source/Energieträger der Heizung', options=['<Please select>','Oil/Öl','Geothermal/Erdwärme','Solar','Wood pellets/Holzpellets','Gas','Steam district heating/Fernwärme-Dampft','Distant district heating/Fernwärme','Coal/Kohle','Light natural gas/Erdgas leicht','Heavy natural gas/Erdgas schwer','LPG/Flüssiggas','Wood/Holz','Wood chips/Holz-Hackschnitzel','Local district heating/Nahwärme','Delivery/Wärmelieferung','Eletricity/Strom'])
                _heating_energy_source = selectbox_to_simplified_german(_heating_energy_source)

                _kennwert = col2.text_input("Power/Kennwert (kW h/(m²a))", value="", max_chars = 5)
                if _kennwert != '':
                    _kennwert = 'V: ' + str(_kennwert) +'kW h/(m²a)'
                else:
                    _kennwert = np.nan

                _building_year = col2.text_input(label='Building construction year/Baujahr des Gebäudes', value="", max_chars = 4)
                _energy_efficiency = col3.selectbox(label='Energy efficiency class/Energieeffizienzklasse', options=['<Please select>','A+','A','B','C','D','E','F','G','H'])
                if _energy_efficiency != '<Please select>':
                    _energy_efficiency = 'Energieeffizienzklasse ' + _energy_efficiency
                else:
                    _energy_efficiency = np.nan

                _energy = ", ".join([item for item in [_energy_certificate, _kennwert, _heating_energy_source, _building_year, _energy_efficiency] if item == item])


            with st.container():
                st.subheader("""
                        \n
                        Utils
                        """)
                col1, col2, col3 = st.columns(3)
                _heating = col1.selectbox(label='Heating/Heizung', options=['<Please select>','Central heating/Zentralheizung','Gas heating/Gasheizung', 'Furnace heating/Ofenheizung', 'District heating/Fernwärme', 'Coal oven/Kohleofen', 'Night storage heating/Nachtspeicherofen'], index=0)
                try:
                    _heating = _heating.split('/')[1]
                except IndexError:
                    _heating = np.nan



                _internet = col1.multiselect(label='Internet', options=['DSL', 'Flatrate', 'WLAN'], default=None)
                _internet = ', '.join([item for item in _internet])

                _internet_speed = col1.selectbox(label='DSL-Speed', options=['<Please select>','7-10 Mbit/s','11-16 Mbit/s','17-25 Mbit/s','26-50 Mbit/s','50-100 Mbit/s','Faster than/schneller als 100 Mbit/s'], index=0)
                _internet_speed = 'schneller als 100 Mbit/s' if _internet_speed == 'Faster than/schneller als 100 Mbit/s' else '' if _internet_speed == '<Please select>' else _internet_speed

                if _internet_speed != '':
                    _internet = _internet + ' ' + _internet_speed

                _furniture = col2.multiselect(label='Furniture/Einrichtung', options=['<Please select>','Furnished/Möbliert', 'Partly furnished/Teilmöbliert'])
                _furniture = ", ".join([item.split('/')[1] for item in _furniture])

                _floor_type = col2.multiselect(label='Floor type/Bodenbelag', options=['Floorboards/Dielen', 'Parquet/Parkett', 'Laminate/Laminat', 'Carpet/Teppich', 'Tiles/Fliesen', 'PVC', 'Underfloor heating/Fußbodenheizung'], default=None)
                _floor_type = ", ".join([item.split('/')[1] for item in _floor_type])

                _tv = col3.multiselect(label='TV', options=['Cable TV/Kabel TV', 'Satellite TV/Satellit TV'], default=None)
                _tv = [item.split('/')[1] for item in _tv]
                _tv = _tv if len(_tv) > 0 else np.nan

                _extras = col3.multiselect(label='Miscellaneous/Sonstiges', options=['Washing machine/Waschmaschine', 'Dishwasher/Spülmaschine', 'Terrace/Terrasse', 'Balcony/Balkon', 'Garden/Garten', 'Shared garden/Gartenmitbenutzung', 'Basement/Keller', 'Elevator/Aufzug', 'Pets allowed/Haustiere erlaubt', 'Bicycle storage/Fahrradkeller'], default=None)
                _extras = ", ".join([item.split('/')[1] for item in _extras])

            "---"
            submitted_form = st.form_submit_button("Submit")




    if submitted_form:
        ###############################
        ### Aggregate inputted info ###
        ###############################

        #############################
        ### Obtain main ads table ###
        #############################
        # Copying is needed to prevent subsequent steps from modifying the cached result from get_original_data()
        ads_df = get_data_from_db(filter_ad_type = 'WG').copy()

        #### Checking inputted info is correct format
        if _city == "<Please select>":
            st.error(f"""
                        Selecting a city is mandatory for analysis.
                        """, icon="🚨")
        elif _street_number == "Str 15":
            st.error(f"""
                        An address is mandatory for analysis.
                        """, icon="🚨")
        elif _zip_code == "12345":
            st.error(f"""
                        The zip code is mandatory for analysis.
                        """, icon="🚨")
        elif not _zip_code.isnumeric():
            st.error(f"""
                        Zip code must containg only numbers.
                        """, icon="🚨")
        elif _price_euros == 0:
            st.error(f"""
                        The warm rent price is mandatory for analysis.
                        """, icon="🚨")
        elif _price_euros > 2000 or _price_euros < 50:
            st.error(f"""
                        The warm rent price is too extreme.
                        """, icon="🚨")
        elif _size_sqm > 60 or _size_sqm < 5:
            st.error(f"""
                        The room size is too extreme.
                        """, icon="🚨")
        elif _size_sqm == 0:
            st.error(f"""
                        The room size is mandatory for analysis.
                        """, icon="🚨")

        else:
            full_address = _street_number + ', ' + _zip_code + ', ' + _city

            with st.spinner(f'Processing address.'):
                lat, lon = geocoding_address(full_address)


            if lat == lat and lon == lon: # Check if lat and lon are not nan
                st.success('Address processed successfully!', icon="✅")

                info_flat = pd.DataFrame.from_dict({
                'id': ['test'],
                'url': ['test'],
                'type_offer': ['WG'],
                'landlord_type': ['Private'],
                'title': ['test'],
                'price_euros': [int(_price_euros)],
                'size_sqm': [float(_size_sqm)],
                'available_rooms': [1],
                'WG_size': [1+int(_male_flatmates)+int(_female_flatmates)+int(_diverse_flatmates)],
                'available_spots_wg': [1],
                'male_flatmates': [int(_male_flatmates)],
                'female_flatmates': [int(_female_flatmates)],
                'diverse_flatmates': [int(_diverse_flatmates)],
                'published_on': [str(time.strftime(f"%d.%m.%Y", time.localtime()))],
                'published_at': [int(time.strftime(f"%H", time.localtime()))],
                'address': [str(full_address)],
                'city': [str(_city)],
                'crawler': ['WG-Gesucht'],
                'latitude': [float(lat)],
                'longitude': [float(lon)],
                'available from': [str(time.strftime(f"%d.%m.%Y", time.localtime()))],
                'available to': [np.nan],
                'details_searched': [True],
                'cold_rent_euros': [int(_cold_rent_euros) if _cold_rent_euros != 0 else np.nan],
                'mandatory_costs_euros': [int(_mandatory_costs_euros) if _mandatory_costs_euros != 0 else np.nan],
                'extra_costs_euros': [int(_extra_costs_euros) if _extra_costs_euros != 0 else np.nan],
                'transfer_costs_euros': [int(_transfer_costs_euros) if _transfer_costs_euros != 0 else np.nan],
                'deposit': [int(_deposit) if _deposit != 0 else np.nan],
                'zip_code': [int(_zip_code)],
                'home_total_size': [int(_home_total_size) if _home_total_size != 0 else np.nan],
                'smoking': [str(_smoking) if _smoking == _smoking else np.nan],
                'wg_type': [str(_wg_type) if len(_wg_type) > 0 else np.nan],
                'languages': [str(_languages) if len(_languages) > 0 else np.nan],
                'age_range': [str(_age_range) if _age_range != '' else np.nan],
                'gender_search': [str(_gender_search)],
                'energy': [str(_energy) if _energy != 'V: kW h/(m²a), ' else np.nan],
                'wg_possible': [np.nan],
                'building_type': [str(_building_type) if _building_type != '' else np.nan],
                'building_floor': [str(_floor) if _floor == _floor else np.nan],
                'furniture': [str(_furniture) if len(_furniture) > 0 else np.nan],
                'kitchen': [np.nan],
                'shower_type': [np.nan],
                'TV': [str(_tv) if _tv == _tv else np.nan],
                'floor_type': [str(_floor_type) if len(_floor_type) > 0 else np.nan],
                'heating': [str(_heating) if _heating == _heating else np.nan],
                'public_transport_distance': [str(_distance_public_transport) if _distance_public_transport != '0 Minuten zu Fuß entfernt' else np.nan],
                'internet': [str(_internet) if str(_internet) != '' else np.nan],
                'parking': [str(_parking) if _parking == _parking else np.nan],
                'extras': [str(_extras) if len(_extras) > 0 else np.nan],
                'Schufa_needed': [True if _schufa_needed == 'Yes' else np.nan]
                    })

                with st.spinner(f'Obtaining WG room info.'):
                    ad_df_processed = process_ads_tables(input_ads_df = info_flat, save_processed = False, df_feats_tag = 'city')
                    st.success('WG room info obtained!', icon="✅")

            else:
                ad_df_processed = None
                st.error(f"""
                        The provided address is invalid: {full_address}
                        """, icon="🚨")


            if ad_df_processed is not None:

        #### Collect files needed for analysis ####

                ### Filter data for analysis ###
                ad_df_analysis_dict = analyse_df_ad(ads_db = ads_df, ad_df = ad_df_processed)


        #### Analysis results ####
                with st.container():
                    st.subheader(f"""
                            This is how your WG compares to other WGs published in the past three months:
                            """)
                    col1, col2, col3, col4 = st.columns(4)

        #### Number ads ####
                    col1.markdown(f"""
                            <font size= "4">**Number of ads posted**</font>
                            """, unsafe_allow_html=True)

                    col1.markdown(f"""
                            <font size= "4">On average, <span style="color:tomato">**{ad_df_analysis_dict['n_hours_post_city'] if ad_df_analysis_dict['n_days_post_city'] <= 1 else ad_df_analysis_dict['n_days_post_city']}**</span> WG room ads were posted in <span style="color:tomato">**{list(ad_df_processed['city'])[0]}**</span> every {'hour' if ad_df_analysis_dict['n_days_post_city'] <= 1 else 'day'}.

                            <span style="color:tomato">**{ad_df_analysis_dict['n_hours_post_zipcode'] if ad_df_analysis_dict['n_days_post_zipcode'] <= 1 else ad_df_analysis_dict['n_days_post_zipcode']}**</span> WG room ads with the same ZIP code <span style="color:tomato">**{list(ad_df_processed['zip_code'])[0]}**</span> were posted every {'hour' if ad_df_analysis_dict['n_days_post_zipcode'] <= 1 else 'day'}.</font>
                            """, unsafe_allow_html=True)

        #### Size room ####
                    col2.markdown(f"""
                            <font size= "4">**Size of the room**</font>
                            """, unsafe_allow_html=True)

                    col2.markdown(f"""
                            <font size= "4">With <span style="color:tomato">**{list(ad_df_processed['size_sqm'])[0]} sqm**</span>, this WG room is bigger than <span style="color:tomato">**{ad_df_analysis_dict['percent_smaller']} %**</span> of WG rooms in <span style="color:tomato">**{list(ad_df_processed['city'])[0]}**</span> and <span style="color:tomato">**{ad_df_analysis_dict['percent_smaller_zipcode']} %**</span> of WG rooms with the same ZIP code <span style="color:tomato">**{list(ad_df_processed['zip_code'])[0]}**</span>.</font>
                            """, unsafe_allow_html=True)

        #### Price ####
                    col3.markdown(f"""
                            <font size= "4">**WG price**</font>
                            """, unsafe_allow_html=True)

                    col3.markdown(f"""
                            <font size= "4">Costing <span style="color:tomato">**{list(ad_df_processed['price_euros'])[0]} €**</span>, this WG room is more expensive than <span style="color:tomato">**{ad_df_analysis_dict['percent_cheaper']} %**</span> of WG rooms in <span style="color:tomato">**{list(ad_df_processed['city'])[0]}**</span> and <span style="color:tomato">**{ad_df_analysis_dict['percent_cheaper_zipcode']} %**</span> of WG rooms with the same ZIP code <span style="color:tomato">**{list(ad_df_processed['zip_code'])[0]}**</span>.</font>
                            """, unsafe_allow_html=True)

        #### Factors influencing price ####
                    col4.markdown(f"""
                            <font size= "4">**Possible factors affecting price**</font>
                            """, unsafe_allow_html=True)


                    def generate_text_possible_price_factors():
                        prompts = ['\n','- Room is large' if ad_df_analysis_dict['percent_smaller_zipcode'] >= 70 else '- Room is small' if ad_df_analysis_dict['percent_smaller_zipcode'] <= 30 else '' +\
                            '- WG in pricier neighborhood' if ad_df_analysis_dict['zip_is_more'] == 'pricier' else '- WG in cheaper neighborhood' if ad_df_analysis_dict['zip_is_more'] == 'cheaper' else '',

                            '- WGs in ' + ad_df_analysis_dict['building_type'] + ' building type tend to be pricier' if ad_df_analysis_dict['building_type'] == 'Neubau' or ad_df_analysis_dict['building_type'] == 'Hochhaus' else '- WGs in ' + ad_df_analysis_dict['building_type'] + ' building type tend to be cheaper' if ad_df_analysis_dict['building_type'] == 'Einfamilienhaus' else '',

                            '- Students WG type tend to be cheaper' if ad_df_analysis_dict['wg_type_studenten'] else '',

                            '- Business WG type tend to be pricier' if ad_df_analysis_dict['wg_type_business'] else '',

                            '- WGs that require Schufa tend to be pricier' if ad_df_analysis_dict['schufa_needed'] else '',

                            '- WGs with companies as landlord tend to be pricier' if ad_df_analysis_dict['commercial_landlord'] else '',

                            '- WGs with capacity for ' + str(ad_df_analysis_dict['capacity']) + ' people tend to be pricier' if ad_df_analysis_dict['capacity'] >= 5 else '- WGs for only 2 people tend to be cheaper' if ad_df_analysis_dict['capacity'] == 2 else '',

                            '- Short-term rental WGs (<30 days) tend to be cheaper' if ad_df_analysis_dict['days_available'] <= 30 else '- WGs with open-end rental time availability tend to be cheaper' if ad_df_analysis_dict['days_available'] > 540 else '']

                        return '\n'.join(text for text in prompts if text != '')

                    col4.markdown(f"""
                                    <font size= "4">{generate_text_possible_price_factors()}</font>
                                    """, unsafe_allow_html=True)


        #### Price prediction ####
                if ad_df_analysis_dict['ad_evaluation'] is not None:
                    '\n'
                    '\n'
                    st.markdown(f"""
                            <font size= "6">**Rent price fairness**</font>
                            """, unsafe_allow_html=True)
                    with st.container():

                        if ad_df_analysis_dict['ad_evaluation'] == 'High price':
                            ad_evaluation = f"""After taking the margin of error in consideration, we consider this warm rent price to be significantly above the price predicted by our AI predictor. Therefore the offered price is in our opinion <span style="color:tomato">**HIGH**</span>."""
                        elif ad_df_analysis_dict['ad_evaluation'] == 'Fairly priced':
                            ad_evaluation = f"""After taking the margin of error in consideration, we consider this warm rent price to be in accordance with our AI precitor prediction. Therefore the offered price is in our opinion <span style="color:tomato">**FAIRLY PRICED**</span>."""
                        elif ad_df_analysis_dict['ad_evaluation'] == 'Low price':
                            ad_evaluation = f"""After taking the margin of error in consideration, we consider this warm rent price to be significantly below the price predicted by our AI predictor. Therefore the offered price is in our opinion <span style="color:tomato">**LOW**</span>."""

                        # Display predictions
                        st.markdown(f"""
                                    The predicted warm rent for this offer is: <span style="color:tomato">**{ad_df_analysis_dict['warm_rent_pred']} €**</span>. This prediction is composed of the predicted cold rent ({ad_df_analysis_dict['cold_rent_pred']} €), plus invariant mandatory and extra costs taken from the ad page. In the ad page, this WG room is priced at <span style="color:tomato">**{int(ad_df_processed['price_euros'])} €**</span> (warm rent). {ad_evaluation}.
                                    """, unsafe_allow_html=True)

                        st.markdown(f"""
                                <font size= "2">*There are two types of rent prices: cold and warm rent. Cold rent is the basic price of the rent, while warm rent usually includes the cold rent, water, heating and house maintenance costs. Warm rent may also include internet and TV/radio/internet taxes and other invariant costs.</font>
                                """, unsafe_allow_html=True)


        #### WG room recommendations ####
                with st.container():
                    '\n'
                    '\n'
                    st.markdown(f"""
                            <font size= "6">**Similar WG rooms**</font>
                            """, unsafe_allow_html=True)
                    with st.spinner(f'Obtaining similar WG rooms.'):

                        ## Filter city
                        ads_df_recommendation = ads_df[ads_df['city'] == list(ad_df_processed['city'])[0]]

                        ## Filter distance
                        lat_ad = float(list(ad_df_processed['latitude'])[0])
                        lon_ad = float(list(ad_df_processed['longitude'])[0])

                        for _day in [5,4,3,2,1]:
                            for _price in [0.3,0.2,0.1,0.05]:
                                for _square in [12000,8000,4000,2000,1000,500]:
                                    if len(ads_df_recommendation) > 10 or len(ads_df_recommendation)<3:

                                        lat_max = lat_ad + meters_to_coord(_square, latitude=lat_ad, direction='north')
                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['latitude'] <=lat_max]

                                        lat_min = lat_ad - meters_to_coord(_square, latitude=lat_ad, direction='south')
                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['latitude'] >=lat_min]

                                        lon_max = lon_ad + meters_to_coord(_square, latitude=lat_ad, direction='west')
                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['longitude'] <=lon_max]

                                        lon_min = lon_ad - meters_to_coord(_square, latitude=lat_ad, direction='east')
                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['longitude'] >=lon_min]


                                        ## Filter price
                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['price_euros'] >= int(list(ad_df_processed['price_euros'])[0])*(1-_price)]
                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['price_euros'] <= int(list(ad_df_processed['price_euros'])[0])*(1+_price)]


                                        ## Filter publication date
                                        selected_days_ago = datetime.date.today() + relativedelta(days=-_day)

                                        ads_df_recommendation = ads_df_recommendation[ads_df_recommendation['published_on'] >= pd.to_datetime(selected_days_ago.strftime("%Y-%m-%d"), format = "%Y-%m-%d")]
                                    else:
                                        break



                    if len(ads_df_recommendation) > 0:
                        evaluations = []
                        for _foo in range(1,len(ads_df_recommendation)+1):
                            _df = ads_df_recommendation.head(_foo).tail(1)
                            evaluations.append(analyse_df_ad(ads_db = ads_df, ad_df = _df)['ad_evaluation'])

                        recomm_ads = {}
                        recomm_ads['Ad title'] = list(ads_df_recommendation['title'])
                        recomm_ads['WG address'] = list(ads_df_recommendation['address'])
                        recomm_ads['Price (€)'] = list(ads_df_recommendation['price_euros'].astype('int'))
                        recomm_ads['Room size (sqm)'] = list(ads_df_recommendation['size_sqm'].astype('int'))
                        recomm_ads['Flatmates'] = list(ads_df_recommendation['capacity'].astype('int')-1)
                        recomm_ads['Available from'] = list(ads_df_recommendation['available_from'])
                        recomm_ads['Available until'] = [item if item == item else "Open end" for item in list(ads_df_recommendation['available_to'])]
                        recomm_ads['Our price evaluation'] = evaluations
                        recomm_ads['Link'] = list(ads_df_recommendation['url'])

                        # Create dataframe
                        recomm_ads = pd.DataFrame.from_dict(recomm_ads)


                        recomm_ads
                    else:
                        st.markdown(f"""
                                <font size= "4">We could not find similar WG rooms recently published on wg-gesucht.de.</font>
                                """, unsafe_allow_html=True)



with tab3:
    st.markdown("""
                ### This <span style="color:tomato">**dashboard**</span> contains everything you want to know about WGs in Germany!
                To get started select below the time period, the city, and the type of market of interest and press "Show results".
                """, unsafe_allow_html=True)


    with st.form("data_overview_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        col1.selectbox("Analysis period:", ['Past week','Past month', 'Past three months', 'Past six months'],#, 'Past year'],
                       key="time_period", index=2)
        col2.selectbox("City:", ['Germany'] + sorted(list(dict_city_number_wggesucht.keys())), key="city_filter", index=0)
        col3.selectbox("Market type:", ['All', 'Apartment', 'Single-room flat', 'WG'], key="market_type", index=3)

        "---"
        submitted = st.form_submit_button("Show results")

        if submitted:
            #############################
            ### Obtain main ads table ###
            #############################
            # Copying is needed to prevent subsequent steps from modifying the cached result from get_original_data()
            ads_df = get_data_from_db()

            #### Filter data for analysis
            df_filtered = filter_original_data(df = ads_df,
                                            city = st.session_state["city_filter"],
                                            time_period = st.session_state["time_period"],
                                            market_type_df = st.session_state["market_type"])


            ### Plot price evolution
            with st.container():
                col1, col2, col3 = st.columns([0.05,1,0.05])
                with col2:
                    if st.session_state["city_filter"] == 'Germany':
                        st.markdown(f'''
                                    #### Price evolution of ads published on wg-gesucht.de in the top {len(dict_city_number_wggesucht.keys())} cities in Germany in the {st.session_state["time_period"].lower()}.
                                    ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                                    #### Price evolution of ads published on wg-gesucht.de in {st.session_state["city_filter"]} in the {st.session_state["time_period"].lower()}.
                                    ''', unsafe_allow_html=True)



                    st.markdown("""
                        *Values displayed here are **warm** rental prices that more accurately reflect living costs. Warm rent usually include the cold rent, water, heating and house maintenance costs. It may also include internet and TV/Radio/Internet taxes.
                        """, unsafe_allow_html=True)

                    st.plotly_chart(price_evolution_per_region(df = ads_df,
                                                                target = 'price_euros',
                                                                time_period = st.session_state["time_period"], city = st.session_state["city_filter"]), use_container_width=True)


            ### Plotting ads per market type
            with st.container():
                col1, col2, col3 = st.columns([0.05,1,0.05])
                with col2:
                    if st.session_state["city_filter"] == 'Germany':
                        st.markdown(f'''
                                    #### Number of ads published on wg-gesucht.de in the selected {len(dict_city_number_wggesucht.keys())} cities in Germany in the {st.session_state["time_period"].lower()}
                                    ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                                    #### Number of ads published on wg-gesucht.de in {st.session_state["city_filter"]} in the {st.session_state["time_period"].lower()}
                                    ''', unsafe_allow_html=True)


                    st.plotly_chart(ads_per_region_stacked_barplot(df = df_filtered, time_period = st.session_state["time_period"], city = st.session_state["city_filter"]), use_container_width=True)


            ### Plotting ads per day
            with st.container():
                col1, col2, col3 = st.columns([1,0.05,0.45])
                with col1:
                    st.plotly_chart(ads_per_day_stacked_barplot(df = df_filtered, city = st.session_state["city_filter"], time_period = st.session_state["time_period"],market_type = st.session_state["market_type"]), height=400, use_container_width=True)
                with col3:
                    st.plotly_chart(ads_per_hour_line_polar(df = df_filtered, city = st.session_state["city_filter"], time_period = st.session_state["time_period"],market_type = st.session_state["market_type"]), height=400, use_container_width=True)




            ### Rank of regions rental price
            with st.container():
                col1, col2, col3 = st.columns([0.05,1,0.05])
                with col2:
                    st.markdown(f"""
                    #### Rank of rental prices in {st.session_state["city_filter"]} in the {st.session_state["time_period"].lower()} (€)
                    """, unsafe_allow_html=True)

                    st.pyplot(price_rank_cities(df = filter_original_data(df = ads_df,
                                            city = st.session_state["city_filter"],
                                            time_period = st.session_state["time_period"],
                                            market_type_df = 'All'),

                                            city = st.session_state["city_filter"]))


            ### Map of sqm price around Germany
            with st.container():
                col1, col2, col3 = st.columns([0.05,1,0.05])
                with col2:
                    st.markdown(f"""
                    #### Square-meter prices in Germany in the {st.session_state["time_period"].lower()} (€/m²)
                    """, unsafe_allow_html=True)

                    st_data = st_folium(map_plotting(plotting_df=prepare_data_for_map(ads_df),market_type = st.session_state["market_type"]), width=700, height=500)

                    st.markdown("""
                        *Square-meter prices were calculated using the cold rent and assumes that all people living in a WG pay the same amount.
                        **Regions without a minimum of 3 ads per ZIP code are not displayed.
                        """, unsafe_allow_html=True)


            with st.container():
                st.markdown(f"""
                    #### Driving factors of rental prices
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                    *Several other factors are also relevant for rental price, including the WG structure and the renting conditions. Below I highlight several of these factors based on the analysis of square-meter cold rental prices (€/m²) in {st.session_state["city_filter"]} in the {st.session_state["time_period"].lower()}.
                    """, unsafe_allow_html=True)

                col1, col2 = st.columns([0.5,0.4])
                with col1:
                    st.markdown("""
                        1.1) Business-type WGs generally pay higher, while student-type WGs tend to pay lower rents.
                        """, unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                        1.2) The number of flatmates in a WG often impacts rental prices.
                        """, unsafe_allow_html=True)


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



                col1, col2, col3 = st.columns([0.6,0.33,0.33])
                with col1:
                    st.markdown("""
                    2.1) Renting a WG for less than a month is the cheapest option. Renting for a fixed term is often more expensive than open-end WG offers.
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    2.2) WGs where the presentation of a Schufa is required for renting are generally more expensive.
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown("""
                    2.3) Renting from commercial landlords (companies) strongly relates to higher rent prices.
                    """, unsafe_allow_html=True)


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


                col1, col2, col3 = st.columns([0.1,1,0.1])
                with col2:
                    st.markdown("""
                    3) The type of the building strongly affects WG price. New buildings (Neubau) in particular have the most expensive offers.
                    """, unsafe_allow_html=True)

                    df_foo = df_filtered[df_filtered['building_type'].notna()]
                    st_data = st.pyplot(my_boxplot(df=df_foo,
                                                    x = 'building_type',
                                                    x_title = "",
                                                    transform_type='str',
                                                    x_axis_rotation = 45,
                                                    fig_height = 5,
                                                    order='mean',
                                                    font_scale=1.5))



with tab4:
    st.markdown("""
                ### Here is how well our <span style="color:tomato">**artificial inteligence (AI)**</span> price predictor works for ads posted last week
                """, unsafe_allow_html=True)

    with st.form("inputs", clear_on_submit=False):
        col1, col2 = st.columns(2)
        city_filter = col1.selectbox("City:", ['Germany'] + sorted(list(dict_city_number_wggesucht.keys())), index=0)
        market_type = col2.selectbox("Market type:", ['All', 'Apartment', 'Single-room flat', 'WG'], index=3)

        submitted = st.form_submit_button("Show latest AI predictor analysis")

        if submitted:
            #############################
            ### Obtain main ads table ###
            #############################
            ads_df = get_data_from_db()

            ## Filter city if not 'Germany'
            if city_filter != 'Germany':
                ads_df = ads_df[ads_df['city'] == city_filter]

            ## Filter market_type if not 'All'
            if market_type != 'All':
                ads_df = ads_df[ads_df['type_offer_simple'] == market_type]

            ads_df['published_on'] = pd.to_datetime(ads_df['published_on'])
            ads_df['week_number'] = ads_df['published_on'].apply(lambda x: x.strftime("%Y")) +'W'+ ads_df['published_on'].apply(lambda x: x.strftime("%V"))

            # There's some bug generating data in week 2023W52 that doesn't exist and I haven't figured out how to solve it in a better way
            ads_df = ads_df[ads_df['week_number'] != '2023W52']

            # Finding the latest week present in the database
            week_number = sorted(set(ads_df['week_number']))[-2]

            ## Identify monday of that week and next monday
            monday_week = pd.to_datetime(week_number + '-1', format = "%GW%V-%w")
            next_monday = monday_week + relativedelta(weeks=1)

            ## Filter ads in current week
            ads_df_past_weeks = ads_df[ads_df['published_on'] < monday_week]
            ads_df_current_week = ads_df[ads_df['published_on'] >= monday_week]
            ads_df_current_week = ads_df_current_week[ads_df_current_week['published_on'] < next_monday]

            # Not sure why I have to remove these
            ads_df_current_week = ads_df_current_week[ads_df_current_week['heating'] != 'Kohleofen']
            ads_df_current_week = ads_df_current_week[ads_df_current_week['age_category_searched'] != '60_100']
            ads_df_current_week = ads_df_current_week[ads_df_current_week['age_category_searched'] != '20_20']
            ads_df_current_week = ads_df_current_week[ads_df_current_week['age_category_searched'] != '40_40']
            ads_df_current_week = ads_df_current_week[ads_df_current_week['age_category_searched'] != '60_60']
            ads_df_current_week = ads_df_current_week[ads_df_current_week['age_category_searched'] != '60_20']
            ads_df_current_week = ads_df_current_week[ads_df_current_week['age_category_searched'] != '40_20']


            ## Load weekly trained model that was trained with all data until that week
            trained_model = pickle.load(open(f'{ROOT_DIR}/model/trained_models/PipelineTrained_allcities_price_euros_LogTarget_{week_number}.pkl','rb'))

            ## Predict current week
            ads_df_current_week['predicted_price_euros'] = trained_model.predict(ads_df_current_week)
            ads_df_current_week['predicted_price_euros'] = round(ads_df_current_week['predicted_price_euros'],0)


            # Calculate residuals
            y_pred = ads_df_current_week['predicted_price_euros']
            y_true = ads_df_current_week['price_euros']
            residuals = y_true - y_pred
            ads_df_current_week['residuals'] = residuals


            ### Simplify table
            ads_df_current_week = ads_df_current_week[['id',
                                                       'city',
                                                       'type_offer_simple',
                                                       'zip_code',
                                                       'price_euros',
                                                       'predicted_price_euros',
                                                       'residuals']]


            if city_filter == 'Germany':
                st.markdown(f'''
                            ### Price prediction for ads published on wg-gesucht.de between {monday_week.strftime("%d.%m.%Y")} and {next_monday.strftime("%d.%m.%Y")} in the top {len(dict_city_number_wggesucht.keys())} cities in Germany
                            ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                            ### Price prediction for ads published on wg-gesucht.de between {monday_week.strftime("%d.%m.%Y")} and {next_monday.strftime("%d.%m.%Y")} in {city_filter}
                            ''', unsafe_allow_html=True)


            ### Plot price prediction
            with st.container():
                col1, col2, col3 = st.columns([0.05,1,0.05])
                with col2:
                    st.markdown(f'''
                                #### Price prediction at scale
                                ''', unsafe_allow_html=True)


                    st.markdown(f'''
                                    {len(ads_df_past_weeks)} ads were used for training the AI predictor until week {week_number}. For the analysis below, the warm rental prices for {len(ads_df_current_week)} ads were predicted by the AI predictor.
                                    ''', unsafe_allow_html=True)


                    st.plotly_chart(predicted_vs_actual_prices_ScatterPlot(df = ads_df_current_week), use_container_width=True)


            ### Plot price prediction error boxplot
            with st.container():
                col1, col2, col3 = st.columns([0.05,1,0.05])
                with col2:
                    st.markdown(f'''
                                #### Difference between predicted and actual rental prices (prediction error)
                                ''', unsafe_allow_html=True)
                    st.markdown(f'''
                                The prediction error strongly depends on the location of the ad. At the same time, due to less ads been available for training the AI predictor and due to the higher variability in prices, the prediction error is higher for apartments and single-room flats.
                                ''', unsafe_allow_html=True)


                    st.plotly_chart(prediction_error_boxplot(df = ads_df_current_week), use_container_width=True)


            ### Prediction error fractions
            with st.container():
                col1, col2, col3 = st.columns([0.05,1,0.05])
                with col2:
                    st.markdown(f'''
                                #### Fraction of ads with price prediction error within a given prediction error range
                                ''', unsafe_allow_html=True)
                    st.markdown(f'''
                                Price for the majority of WG ads are predicted within a 50 € range from the actual rental price. The majority of prices for apartments and single-room flats are predicted within a 100 € range.
                                ''', unsafe_allow_html=True)

                    if market_type == 'WG' or market_type == 'All':
                        st.markdown(f'''
                                    #### WGs
                                    ''', unsafe_allow_html=True)
                        st.plotly_chart(fraction_prediction_error_barplot(df = ads_df_current_week,
                                                                        market_type = 'WG'), use_container_width=True)
                    if market_type == 'Single-room flat' or market_type == 'All':
                        st.markdown(f'''
                                #### Single-room flats
                                ''', unsafe_allow_html=True)
                        st.plotly_chart(fraction_prediction_error_barplot(df = ads_df_current_week,
                                                                      market_type = 'Single-room flat'), use_container_width=True)
                    if market_type == 'Apartment' or market_type == 'All':
                        st.markdown(f'''
                                #### Apartments
                                ''', unsafe_allow_html=True)
                        st.plotly_chart(fraction_prediction_error_barplot(df = ads_df_current_week,
                                                                      market_type = 'Apartment'), use_container_width=True)



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
