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
import numpy as np
import plotly.express as px

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


#----Functions------
@st.cache
def get_original_data():
    return get_processed_ads_table()

def filter_original_data(df,city,time_period):
    ## Format dates properly
    df['published_on'] = pd.to_datetime(df['published_on'], format = "%Y-%m-%d")

    ## Filter table
    # City of choice
    if st.session_state["city"] != 'Germany':
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

    return df

def ads_per_city_stacked_barplot(df,time_period):
    city_ads_df = df[['url', 'city',"type_offer_simple"]].groupby(['city',"type_offer_simple"]).count().rename(columns = {'url':'count'}).sort_values(by = ['count'], ascending=False).reset_index()

    st.markdown(f'Total ads published on wg-gesucht.de in 25 cities in Germany in the {time_period.lower()}.', unsafe_allow_html=True)

    fig = px.bar(city_ads_df, x="city", y="count", color="type_offer_simple",
            labels={
                "city": 'City',
                "count": 'Number of ads published',
                "type_offer_simple": "Type of ad"
            },
            template = "ggplot2" #["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
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
    return fig

def ads_per_day_stacked_barplot(df,city,time_period,market_type):
    df_plot = df[['url', 'city', 'published_on']].groupby(['published_on','city']).count().rename(columns={'url':'count'}).sort_values(by = ['published_on'], ascending=True).reset_index()

    foo = df_plot[['count', 'city', 'published_on']].groupby(['published_on']).sum().rename(columns={'count':'mean'})
    mean_ads_day = int(round(foo['mean'].mean(),0))

    st.markdown(f'On average {mean_ads_day} {market_type} ads were published on wg-gesucht.de every day in {city} in the {time_period.lower()}.', unsafe_allow_html=True)

    fig = px.bar(df_plot, x="published_on", y="count", color="city",
                    # title=' ',
            labels={
                "published_on": 'Make this label disappear',
                "count": 'Number of ads published per day',
                "city": "City"
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
    wg_df_time = df[['url', 'day_of_week_publication','published_at']].groupby(['day_of_week_publication','published_at']).count().rename(columns = {'url':'count'}).reset_index()

    plotting_values = [float(i) for i in list(np.arange(0, 360, 360/24))]
    mapping_dict = dict(zip(range(0,24), plotting_values))
    wg_df_time['published_at_radians'] = wg_df_time.published_at.map(mapping_dict)

    days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    st.markdown(f'Average number of {market_type} ads published per hour on each day of the week in {city} in the {time_period.lower()}.', unsafe_allow_html=True)

    fig = px.line_polar(wg_df_time,
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
                      radialaxis_tickvals=[50,100,150])

    return fig








#----Page start------
st.markdown("""
                # Welcome to the <span style="color:tomato">WG prices in Germany</span> dashboard!
                ## Here you'll find everything you want to know about WG prices in Germany.
                For technical details, please check my [GitHub](https://github.com/chvieira2/housing_crawler).
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
placeholder = st.empty()

with placeholder.container():

    with st.form("entry_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        col1.selectbox("Analysis period:", ['Past week','Past month', 'Past three months', 'Past six months', 'Past year'], key="time_period", index=1)
        col2.selectbox("City:", ['Germany'] + sorted(list(dict_city_number_wggesucht.keys())), key="city", index=0)
        col3.selectbox("Market type:", ['WG', 'Single-room flat', 'Apartment'], key="market_type", index=0)

        "---"
        submitted = st.form_submit_button("Show results")

        if submitted:
            #############################
            ### Obtain main ads table ###
            #############################
            # Copying is needed to prevent subsequent steps from modifying the cached result from get_original_data()
            ads_df = get_original_data().copy()

            df_filtered = filter_original_data(df = ads_df,
                                          city = st.session_state["city"],
                                          time_period = st.session_state["time_period"])


            ## Filter type of offer
            wg_df = df_filtered.query('type_offer_simple == "WG"').reset_index().drop(columns=['index'])

            singleroom_df = df_filtered.query('type_offer_simple == "Single-room flat"').reset_index().drop(columns=['index'])

            flathouse_df = df_filtered.query('(type_offer_simple == "Apartment")').reset_index().drop(columns=['index'])



            ###############################################
            ### Creates the different tabs with results ###
            ###############################################
            tab1, tab2, tab3 = st.tabs(["Global analysis", "Price analysis", "Owl"])

            with tab1:
                st.header(f"""
                    #### A global view into ads published on wg-gesucht.de in {st.session_state["city"]} in the {st.session_state["time_period"].lower()}
                    """)


                ### Plotting ads per market type
                if st.session_state["city"] == 'Germany':
                    placeholder = st.empty()
                    with placeholder.container():
                        col1, col2, col3 = st.columns([0.05,1,0.05])
                        with col2:
                            st.plotly_chart(ads_per_city_stacked_barplot(df = df_filtered, time_period = st.session_state["time_period"]), use_container_width=True)


                ### Plotting ads per day
                placeholder = st.empty()
                with placeholder.container():
                    col1, col2, col3 = st.columns([1,0.05,0.45])
                    with col1:
                        st.plotly_chart(ads_per_day_stacked_barplot(df = wg_df, city = st.session_state["city"], time_period = st.session_state["time_period"],market_type = st.session_state["market_type"]), height=400, use_container_width=True)
                    with col3:
                        st.plotly_chart(ads_per_hour_line_polar(df = wg_df, city = st.session_state["city"], time_period = st.session_state["time_period"],market_type = st.session_state["market_type"]), height=400, use_container_width=True)

            with tab2:
                st.header("Price analysis")
                st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

            with tab3:
                st.header("An owl")
                st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
