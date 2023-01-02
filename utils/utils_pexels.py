# Author: Arturo Aguilar Lagunas, see https://github.com/AguilarLagunasArturo/pexels-image-downloader.git
# Modified by Yuyang Li

import logging
import os
import sys
from time import sleep

import requests
import ulid
from pexels_api import API

def get_api(cred):
    return API(cred)

def download_pexels(query, api, basedir, n_photos=40):
    meta = []
    
    do_sleep = False
    
    # Search photos
    logging.info("Searching: {}".format(query))
    per_page = n_photos if n_photos < 80 else 80
    
    try:
        api.search(query, per_page)
        logging.info("Total results: {}".format(api.total_results))
    
        if not api.json["photos"]: return meta, True
        if n_photos > api.total_results:
            n_photos = api.total_results
            logging.info("Not enough photos, downloading {} photos instead".format(n_photos))
            
        # Create directory if does not exists
        path = os.path.join(basedir, query.replace(" ", "-"))
        os.makedirs(path, exist_ok=True)
        
        logging.info("Writing to main folder: {}".format(path))
        
        all_entires = api.get_entries()[:n_photos]
    except:
        do_sleep = True
        all_entires = []
    
    # Get photos
    for photo in all_entires:
        dir = path
        filename = f"pexels-{photo.id}"
            
        download_url = photo.original
        # download_url = photo.large2x
        filename += "." + photo.extension if not photo.extension == "jpeg" else ".jpg"
            
        photo_path = os.path.join(dir, filename)
        with open(photo_path, "wb") as f:
            try:
                f.write(requests.get(download_url, timeout=15).content)
                meta.append({
                    "id": ulid.new().str,
                    "query": query,
                    "pid": photo.id,
                    "source": "Pexels",
                    "urls": {
                        "landscape": photo.landscape,
                        "large": photo.large,
                        "large2x": photo.large2x,
                        "medium": photo.medium,
                        "original": photo.original,
                        "portrait": photo.portrait,
                        "small": photo.small,
                        "tiny": photo.tiny,
                        "url": photo.url,
                    },
                    "description": photo.description,
                    "extension": photo.extension,
                    "photographer": photo.photographer,
                    "width": photo.width,
                    "height": photo.height
                })
                sleep(0.5)
            except:
                logging.error("\nInterrupted, {} photos downloaded".format(n_photos-1))
                os.remove(photo_path)
                break
            
    return meta, do_sleep
