# By Chuyue Tang and Yuyang Li

import logging
import os
import sys
import urllib
from time import sleep

import flickrapi
import ulid


def get_api(cred):
    return flickrapi.FlickrAPI(cred['key'], cred['secret'])

def download_flickr(query, api, basedir, n_photos=40):
    do_sleep = False
    meta = []
    final_meta = []
    
    path = os.path.join(basedir, query.replace(" ", "-"))
    os.makedirs(path, exist_ok=True)
    logging.info("Writing to main folder: {}".format(path))
    
    try:
        photos = api.walk(tags=query, content_type='1', extras='url_c', per_page=100)
        for i, photo in enumerate(photos):
            if i >= n_photos:
                break
            meta.append({
                    "id": ulid.new().str,
                    "query": query,
                    "pid": photo.get('id'),
                    "source": "Flickr",
                    "url": photo.get('url_c'),
                    "description": photo.get('title'),
                    "photographer": photo.get('owner'),
                    "width": photo.get('width_c'),
                    "height": photo.get('height_c')
                })
    except:
        do_sleep = True
        
    for i, m in enumerate(meta):
        if (m['url'] is None):
            continue
        filename = f"flickr-{m['pid']}.jpg"
        try:
            urllib.request.urlretrieve(m['url'], os.path.join(path, filename))
            final_meta.append(m)
            sleep(0.5)
        except:
            logging.error("\nInterrupted, {} photos downloaded".format(n_photos-1))
            break
            
    return final_meta, do_sleep
