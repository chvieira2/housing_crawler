## Welcome to [WG-prices](https://wgs-in-germany.streamlit.app/)! 
WG-prices is a free and intuitive webpage where anyone can analyse the market of shared flats (WGs) in Germany. It was created out of curiosity and desire to help people understand the housing market better. With this tool I want to help anyone answer questions like:
- Is the price for this WG fair?
- Are there similar offers that I should consider?
- How likely am I to find a similar offer, and how long would it take?

It's easy and intuitive! Visit the webpage and try it out: [WG-prices](https://wgs-in-germany.streamlit.app/)


## Why do we need WG-prices?
The price paid for a WG is somewhat related to the rental price of the flat. However, the WG market in Germany is saturated in several cities, making people living in WGs susceptible to accept WGs that charge more than they should. This is the case specially for younger adults and people coming from abroad that have little resources and knowledge to judge the fairness of an offer.

WG-prices is 100% free of charge for everyone. If you live or wants to live in a WG, WG-prices helps you judge if the values charged are in accordance to the current market, or if someone is trying to exploit you.


## How to use it?
Open the [link](https://wgs-in-germany.streamlit.app/) and select one of the tabs.
- If you already have an ad from wg-gesucht.de that you would like to investigate, open the tab "Analyse WG from url link", paste the link to the wg-gesucht ad and click "submit";
- If you would like to compare your own flat to the current market, open the tab "Analyse my own WG", insert the information about your WG and click "submit". Information submitted is only used for your analysis and is not saved anywhere;
- If you are instead only interested in a global overview of the current market, select the tab "Overview of the WG market", select the parameters of your choice and click "Show results";
- If you are interested in the predictive model of prices, select the tab "The predictive model of WG prices".


## Behind the curtains
Inside WG-prices I developed a webcrawler tool to collect data from wg-gesucht.de. This tool has been constantly running since August 2022 and I accumulated over 100.000 ads for WGs, single-room, multi-room flats and houses. [WG-prices](https://wgs-in-germany.streamlit.app/) is a platform for anyone to access all of this data without any coding knowledge.

Particularly, I used these data to generate a predictive model of prices. Please check the [wg_price_predictor](https://github.com/chvieira2/wg_price_predictor) repository if you are interested in the predictive model. This model is used to answer a simple question: **given the current market, how much should be charged for a WG with given specifications**. As the model is not perfect, the full analysis of the current market in the past 3 months in Germany is displayed in a dashboard format to help you form your own judgement.


## Finding similar offers
Another important use of this data is to identify similar offers. For that, I collect information on the room of interest (either an ad on wg-gesucht.de or your own) and compare it to the database of ads. Recently published, similar ads are found by identifying offers with similar price and at proximal similar locations. If less than 3 similar ads are found, I extend the search to the days before.


## How to use my webcrawler in your own project
If you are interested in using this webcrawler for your own project, please:
1) Download this repository.
2) In your terminal, go the project folder you just downloaded.
3) If you would like to use it as a module in your own code, install it by running run "pip install -e ."
4) If you just want to run my code directly for a city of interest, simply run "python main.py -l name_city".

For an example on how this crawler could be used in your own project, please see [Livablestreets](https://github.com/chvieira2/livablestreets).


## About
WG-prices was created by [Carlos H. Vieira e Vieira](https://github.com/chvieira2). The code is written in Python, and the app is hosted and published with [Streamlit](https://streamlit.io/). Real estate market ads were scraped from [wg-gesucht.de](https://www.wg-gesucht.de/).
