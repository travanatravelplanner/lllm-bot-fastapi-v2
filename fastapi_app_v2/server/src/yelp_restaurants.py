import os
import json
import requests
import sys
import urllib
import logging

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO)

# For Python 3.0 and later
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.parse import urlencode


API_KEY= os.getenv("YELP_API_KEY")
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'

SEARCH_LIMIT = 20


def request(host, path, api_key, url_params=None):
    url_params = url_params or {}
    url = '{0}{1}'.format(host, quote(path.encode('utf8')))
    headers = {'Authorization': 'Bearer %s' % api_key}
    response = requests.request('GET', url, headers=headers, params=url_params)
    return response.json()


def search(api_key, term, location):
    url_params = {
        'term': term.replace(' ', '+'),
        'location': location.replace(' ', '+'),
        'limit': SEARCH_LIMIT
    }
    try:
        return request(API_HOST, SEARCH_PATH, api_key, url_params=url_params)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_business(api_key, business_id):
    business_path = BUSINESS_PATH + business_id
    return request(API_HOST, business_path, api_key)


def query_api(term, location):
    response = search(API_KEY, term, location)
    businesses = response.get('businesses')

    if not businesses:
        print(u'No businesses for {0} in {1} found.'.format(term, location))
        return

    results = []

    for business in businesses:
        business_id = business['id']
        business_details = get_business(API_KEY, business_id)
        results.append({
            'name': business_details['name'],
            'category': business_details['categories'][0]['title'],
            'display_address': business_details['location']['display_address'],
            'phone': business_details['phone'],
        })

    return results

def main(destination):
    term = "dinner"
    location = str(destination)
    try:
        results = query_api(term, location)
    except HTTPError as error:
        sys.exit(
            'Encountered HTTP error {0} on {1}:\n {2}\nAbort program.'.format(
                error.code,
                error.url,
                error.read(),
            )
        )
    return results