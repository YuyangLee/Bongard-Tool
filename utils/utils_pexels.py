# Author: Arturo Aguilar Lagunas, see https://github.com/AguilarLagunasArturo/pexels-image-downloader.git
# Modified by Yuyang Li

import logging
import os
import sys

import requests
import ulid
from pexels_api import API


def download_pexels(query, api_key, basedir, n_photos=40):
    api = API(api_key)
    meta = []
    
    # Search photos
    logging.info("Searching: {}".format(query))
    per_page = n_photos if n_photos < 80 else 80
    api.search(query, per_page)
    logging.info("Total results: {}".format(api.total_results))
    
    if not api.json["photos"]: return meta
    if n_photos > api.total_results:
        n_photos = api.total_results
        logging.info("Not enough photos, downloading {} photos instead".format(n_photos))
        
    # Create directory if does not exists
    path = os.path.join(basedir, query.replace(" ", "-"))
    os.makedirs(path, exist_ok=True)
    
    logging.info("Writing to main folder: {}".format(path))
    
    # Get photos
    for photo in api.get_entries()[:n_photos]:
        filename = str(n_photos).zfill(len(str(n_photos)))

        dir = path
        filename = f"pexels-{photo.id}"
        if not os.path.isdir(dir):
            os.mkdir(dir)
            
        download_url = photo.large2x
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
            except:
                logging.error("\nInterrupted, {} photos downloaded".format(n_photos-1))
                os.remove(photo_path)
                if not os.listdir(dir):
                    os.rmdir(dir)
                    if not os.listdir(os.path.split(dir)[0]):
                        os.rmdir(os.path.split(dir)[0])
                break
            
        if not api.search_next_page(): break

    return meta
