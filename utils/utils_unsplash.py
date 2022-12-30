# Modified by Yuyang Li

import logging
import os
import sys
import urllib
from time import sleep

import requests
import ulid
from unsplash.api import Api
from unsplash.auth import Auth


def get_api(cred):
    key, secret = cred['key'], cred['secret']
    return Api(Auth(key, secret, redirect_uri="urn:ietf:wg:oauth:2.0:oob"))

def download_unsplash(query, api, basedir, n_photos=40):
    meta = []
    
    do_sleep = False
    
    # Search photos
    logging.info("Searching: {}".format(query))
    per_page = n_photos if n_photos < 40 else 40
    
    try:
        results = api.search.photos(query, per_page=per_page)['results']
        logging.info("Total results: {}".format(len(results)))
    
        if n_photos > len(results):
            n_photos = len(results)
            logging.info("Not enough photos, downloading {} photos instead".format(n_photos))
            
        # Create directory if does not exists
        path = os.path.join(basedir, query.replace(" ", "-"))
        os.makedirs(path, exist_ok=True)
        
        logging.info("Writing to main folder: {}".format(path))
        
        all_entires = results[:n_photos]
    except:
        do_sleep = True
        all_entires = []
    
    # Get photos
    for photo in all_entires:
        photo_vars = vars(photo)
        dir = path
        filename = f"unsplash-{photo.id}.jpg"
            
        download_url = photo.urls.raw
        download_url = download_url.split('?')[0]
        
        photo_path = os.path.join(dir, filename)
        with open(photo_path, "wb") as f:
            try:
                urllib.request.urlretrieve(download_url, photo_path)
                meta.append({
                    "id": ulid.new().str,
                    "query": query,
                    "pid": photo.id,
                    "description": query,
                    "source": "Unsplash",
                    "urls": download_url,
                    "photographer": photo.user.id,
                    "width": photo.width,
                    "height": photo.height
                })
                
                if 'description' in photo_vars:
                    meta[-1]['description'] = photo_vars['description']
                    
                sleep(1.0)
            except Exception as e:
                print(e)
                logging.error("\nInterrupted, {} photos downloaded".format(n_photos-1))
                break
            
    return meta, do_sleep
