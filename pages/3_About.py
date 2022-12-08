# -*- coding: utf-8 -*-

"""Housing Crawler - search for flats by crawling property portals and save them locally.

This is the dashboard/app implementation of the analysis of ads obtained from wg-gesucht.de"""

__author__ = "Carlos Henrique Vieira e Vieira"
__version__ = "1.0"
__maintainer__ = "chvieira2"
__email__ = "carloshvieira2@gmail.com"
__status__ = "Production"


import streamlit as st


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

#----Content starts------
placeholder_map = st.empty()
with placeholder_map.container():


    st.write('\n')
    st.markdown("""
            ### What is WG-prices?
            WG-prices is a intuitive webpage where anyone can visualize the WG market in Germany.
            WG-prices was created by [chvieira2](https://github.com/chvieira2) out of curiosity and desire to help others. Its purpose is to help people understand the housing market better, in particular the WG market.

            ### Why do we need WG-prices?
            The price paid for a WG is theoratically bound to the rental price of the flat. However, the WG market is saturated, making people living in WGs susceptible to accept offers that charge more than they should. This is the case specially for younger adults and people coming from abroad that have little resources to judge the fairness of an offer.

            ### Who is WG-prices meant for?
            WG-prices is 100% free for everyone to use it. If you live or wants to live in a WG, WG-prices helps you judge if the values charged are in accordance to the current market, or if someone is trying to exploit you. If you own a flat and is considering renting a room, WG-prices helps you decide a fair price in accordance to the current market.

            ### How does WG-prices work?
            WG-prices collects data from wg-gesucht.de, analysis it and uses it to generate a predictive model of prices. This model is used to answer a simple question: given the current market, how much should be charged for a WG with these specifications. As the model is not perfect, the full analysis of the current market in the past 3 months in Germany is displayed in a dashboard format to help you form your own judgement.
            """, unsafe_allow_html=True)
