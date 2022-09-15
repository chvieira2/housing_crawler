#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Housing Crawler - search for flats by crawling property portals and save them locally.

This is the main command-line executable, for running on the console.
For long-term search, please run CrawlWgGesucht().long_search() inside housing_crawler/crawl_wggesucht.py instead"""

import argparse
import os
# import logging
# import time
# from pprint import pformat

from housing_crawler.crawl_wggesucht import CrawlWgGesucht

__author__ = "Carlos Henrique Vieira e Vieira"
__version__ = "1.0"
__maintainer__ = "chvieira2"
__email__ = "carloshvieira2@gmail.com"
__status__ = "Production"

# # init logging
# if os.name == 'posix':
#     # coloring on linux
#     CYELLOW = '\033[93m'
#     CBLUE = '\033[94m'
#     COFF = '\033[0m'
#     LOG_FORMAT = '[' + CBLUE + '%(asctime)s' + COFF + '|' + CBLUE + '%(filename)-18s' + COFF + \
#                  '|' + CYELLOW + '%(levelname)-8s' + COFF + ']: %(message)s'
# else:
#     # else without color
#     LOG_FORMAT = '[%(asctime)s|%(filename)-18s|%(levelname)-8s]: %(message)s'
# logging.basicConfig(
#     format=LOG_FORMAT,
#     datefmt='%Y/%m/%d %H:%M:%S',
#     level=logging.INFO)
# __log__ = logging.getLogger('housing_crawler')


def launch_search(location_name = 'berlin', page_number = 3,
                    filters = ["wg-zimmer"], path_save = None):
    """Starts the crawler"""
    crawler = CrawlWgGesucht()
    crawler.crawl_all_pages(location_name = location_name, page_number = page_number,
                    filters = filters, path_save = path_save)


def main():
    """Processes command-line arguments, loads the config, launches the housing_crawler"""
    parser = argparse.ArgumentParser(
        description=("Searches for flats on wg-gesucht.de"),
        epilog="Designed by chvieira2"
    )

    parser.add_argument('--location', '-l',
                        action='store',
                        type=str,
                        default='berlin',
                        help=f'Name of city to search for flats'
                        )

    parser.add_argument('--pages', '-p',
                        action='store',
                        type=int,
                        default=3,
                        help=f'Number of pages to crawl. Firt few pages contain most recent posted ads. Increasing this number risks getting caught by reCAPTCH.'
                        )

    parser.add_argument('--filters', '-f',
                        action='store',
                        type=list,
                        default=["wg-zimmer"],
                        help=f'''List of searching filters. Input format = ["filter1","filter2"]
                                "wg-zimmer" = Room search in shared flat
                                "1-zimmer-wohnungen" = 1 room flat
                                "wohnungen" = Apartments to rent
                                "haeuser" = Houses to rent
                                '''
                        )

    parser.add_argument('--savepath', '-s',
                        action='store',
                        type=str,
                        default=None,
                        help=f'''Destination path where the files should be saved
                                '''
                        )
    args = parser.parse_args()

    # start hunting for flats
    launch_search(location_name=args.location,
                     page_number=args.pages,
                     filters=args.filters,
                     path_save = args.savepath)


if __name__ == "__main__":
    main()
