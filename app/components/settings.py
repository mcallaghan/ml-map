import os

title='Climate policy instruments'

url_title = title.lower().replace(' ','-')

url_base = os.getenv('CHA_BASEURL', '/')
url_req = os.getenv('CHA_URL_REQUESTS', url_title)
url_routes = os.getenv('CHA_URL_ROUTES', url_title)
url_asset = os.getenv('CHA_URL_ASSET', '/')