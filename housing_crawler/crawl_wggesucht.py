import re
import time
import requests
# import logging
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from housing_crawler.abstract_crawler import Crawler
from housing_crawler.string_utils import remove_prefix
from housing_crawler.utils import save_file

from config.config import ROOT_DIR



class CrawlWgGesucht(Crawler):
    """Implementation of Crawler interface for WgGesucht"""

    # __log__ = logging.getLogger('housing_crawler')

    def __init__(self):
        self.base_url = 'https://www.wg-gesucht.de/'

        # logging.getLogger("requests").setLevel(logging.WARNING)
        self.existing_findings = []

        # The number after the name of the city is specific to the city
        self.dict_city_number = {
            'Berlin': '8',
            'Muenchen': '90',
            'Stuttgart': '124',
            'Koeln': '73',
            'Hamburg': '55',
            'Duesseldorf': '30',
            'Bremen':'17',
            'Leipzig':'77',
            'Kiel':'71',
            'Heidelberg':'59',
            'Karlsruhe':'68',
            'Hannover':'57'
        }

    def url_builder(self, location_name, page_number,
                    filters):

        # Make sure that the city name is correct
        location_name = location_name.capitalize()

        filter_code = []

        if "wg-zimmer" in filters:
            filter_code.append('0')
        if "1-zimmer-wohnungen" in filters:
            filter_code.append('1')
        if "wohnungen" in filters:
            filter_code.append('2')
        if "haeuser" in filters:
            filter_code.append('3')

        filter_code = '+'.join(filter_code)
        filter_code = ''.join(['.']+[filter_code]+['.1.'])


        return self.base_url +\
            "-und-".join(filters) +\
                '-in-'+ location_name + '.' + self.dict_city_number.get(location_name) +\
                    filter_code + str(page_number) + '.html'

    def get_soup_from_url(self, url, max_sleep_time = 5):
        """
        Creates a Soup object from the HTML at the provided URL

        Overwrites the method inherited from abstract_crawler. This is
        necessary as we need to reload the page once for all filters to
        be applied correctly on wg-gesucht.
        """

        # Sleeping for random seconds to avoid overload of requests
        sleeptime = np.random.uniform(2, max_sleep_time)
        print(f"sleeping for: {round(sleeptime,2)} seconds")
        time.sleep(sleeptime)

        # Setup an agent
        self.rotate_user_agent()

        sess = requests.session()
        # First page load to set filters; response is discarded
        sess.get(url, headers=self.HEADERS)
        # Second page load
        print(f"Crawling {url}")
        resp = sess.get(url, headers=self.HEADERS)
        print(f"Got response code {resp.status_code}")

        # Return soup object
        return BeautifulSoup(resp.content, 'html.parser')

    def extract_data(self, soup):
        """Extracts all exposes from a provided Soup object"""

        # Find ads
        findings = soup.find_all(lambda e: e.has_attr('id') and e['id'].startswith('liste-'))
        findings_list = [
        e for e in findings if e.has_attr('class') and not 'display-none' in e['class']
        ]
        print(f"Extracted {len(findings_list)} entries")
        return findings_list

    def parse_urls(self, location_name, page_number, filters):
        """Parse through all exposes in self.existing_findings to return a formated dataframe.
        """
        # Process city name to match url
        location_name = location_name.lower().replace(' ', '_')\
            .replace('ã','a').replace('õ','o')\
            .replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')\
            .replace('ç','c')\
            .replace('à','a').replace('è','e').replace('ì','i').replace('ò','o').replace('ù','u')\
            .replace('â','a').replace('ê','e').replace('î','i').replace('ô','o').replace('û','u')\
            .replace('ä','ae').replace('ë','e').replace('ï','i').replace('ö','oe').replace('ü','ue')\
            .replace('ß','ss').replace('ñ','n')\
            .replace('ī','i').replace('å','a').replace('æ','ae').replace('ø','o').replace('ÿ','y')\
            .replace('š','s').replace('ý','y')\
            .replace('ş','s').replace('ğ','g')
        location_name = location_name.capitalize()

        print(f'Processed location name. Searching for: {location_name}')


        # Create list with all urls for crawling
        list_urls = [self.url_builder(location_name = location_name, page_number = page,
                    filters = filters) for page in range(page_number)]

        print(f'Created {len(list_urls)} urls for crawling')

        # Crawling each page and adding findings to
        for url in list_urls:
            soup = self.get_soup_from_url(url)
            new_findings = self.extract_data(soup)
            if len(new_findings) == 0:
                print('====== Stopping retrieving pages. Stuck at Recaptcha ======')
                break
            self.existing_findings = self.existing_findings + (new_findings)

    def crawl_all_pages(self, location_name, page_number,
                    filters = ["wg-zimmer","1-zimmer-wohnungen","wohnungen","haeuser"]):

        # Obtaining pages
        if len(self.existing_findings) == 0:
            self.parse_urls(location_name = location_name, page_number = page_number,
                    filters = filters)

        # Extracting info of interest from pages
        print(f"Crawling {len(self.existing_findings)} ads")
        entries = []
        for row in self.existing_findings:
            # Ad title
            title_row = row.find('h3', {"class": "truncate_title"})
            title = title_row.text.strip()

            # Ad url
            ad_url = self.base_url + remove_prefix(title_row.find('a')['href'], "/")

            # Ad image link
            # image = re.match(r'background-image: url\((.*)\);', row.find('div', {"class": "card_image"}).find('a')['style'])[1]

            # Room details and address
            detail_string = row.find("div", {"class": "col-xs-11"}).text.strip().split("|")
            details_array = list(map(lambda s: re.sub(' +', ' ',
                                                    re.sub(r'\W', ' ', s.strip())),
                                    detail_string))
            rooms_tmp = re.findall(r'\d Zimmer', details_array[0])
            rooms = rooms_tmp[0][:1] if rooms_tmp else 0
            address = details_array[2] + ', ' + details_array[1]

            # Flatmates
            try:
                flatmates = row.find("div", {"class": "col-xs-11"}).find("span", {"class": "noprint"})['title']
                flatmates_list = [int(n) for n in re.findall('[0-9]+', flatmates)]
            except TypeError:
                flatmates_list = [0,0,0,0]

            # Price
            numbers_row = row.find("div", {"class": "middle"})
            price = numbers_row.find("div", {"class": "col-xs-3"}).text.strip()

            # Size and ad dates
            dates = re.findall(r'\d{2}.\d{2}.\d{4}',
                            numbers_row.find("div", {"class": "text-center"}).text)

            size = re.findall(r'\d{1,4}\sm²',
                            numbers_row.find("div", {"class": "text-right"}).text)
            if len(size) == 0:
                size=['0']

            if len(size) == 0:
                size = [0]

            details = {
                'id': int(ad_url.split('.')[-2]),
                # 'image': image,
                'url': ad_url,
                'title': title,
                'price(euros)': price.split(' ')[0],
                'size(sqm)': int(re.findall('[0-9]+', size[0])[0]),
                'available rooms': rooms,
                'WG size': flatmates_list[0],
                'available spots': flatmates_list[0]-flatmates_list[1]-flatmates_list[2]-flatmates_list[3],
                'male flatmates': flatmates_list[1],
                'female flatmates': flatmates_list[2],
                'diverse flatmates': flatmates_list[3],
                'address': address,
                'crawler': 'WG-Gesucht'
            }
            if len(dates) == 2:
                details['from'] = dates[0]
                details['to'] = dates[1]
            elif len(dates) == 1:
                details['from'] = dates[0]
                details['to'] = 'open end'

            entries.append(details)

            # self.__log__.debug('extracted: %s', entries)

        df = pd.DataFrame(entries)
        save_file(df = df, file_name=f'{location_name}.csv', local_file_path=f'housing_crawler/data/{location_name}')


        return df

if __name__ == "__main__":
    test = CrawlWgGesucht()
    print(test.crawl_all_pages(location_name = 'Berlin', page_number = 20,
                    filters = ["wg-zimmer"]))
