import json
import logging
import os
import sqlite3
from datetime import datetime
from time import sleep

import tensorboardX
from tqdm import tqdm

from utils.utils_dev import get_yaml_data
from utils.utils_pexels import download_pexels
from utils.utils_flickr import download_flickr

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--logdir", type=str, default="logs")
parser.add_argument("--source", type=str, default="flickr")

args = parser.parse_args()

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=f'{args.logdir}/pexels-{ str(datetime.now().timestamp()) }.log', format=LOG_FORMAT)

cred_path = "credentials/credentials.yaml"
query_path = "data/queries.json"

metadata_path = "dataset/metadata.json"
dataset_basedir = "dataset/images"

match args.source:
    case "pexels":
        cred = get_yaml_data(cred_path)['pexels']
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        download = download_pexels
        from utils.utils_pexels import get_api
    case "flickr":
        download = download_flickr
        cred = get_yaml_data(cred_path)['flickr']
        os.environ['HTTP_PROXY']="http://127.0.0.1:7890"
        os.environ['HTTPS_PROXY']="http://127.0.0.1:7890"
        from utils.utils_flickr import get_api
    case _:
        raise NotImplementedError()
        
hour_amount, daily_amount = cred['maxPerHour'], cred['maxPerDay']
tokens = iter(cred['tokens'])

os.makedirs(dataset_basedir, exist_ok=True)

if os.path.exists(metadata_path):
    metadata = json.load(open(metadata_path, "r"))
else:
    metadata = { 'metadata': [], 'queries': {} }

queries = json.load(open(query_path, "r"))
n_each_query = 10

api_stat = {}

dataset_stat = {}

api = get_api(next(tokens))

hour_count = 0

for query in tqdm(queries):
    meta, do_sleep = download(query, api, dataset_basedir, n_each_query)
    
    metadata['metadata'] += meta
    metadata['queries'][query] = len(meta) if query not in metadata['queries'] else metadata['queries'][query] + len(meta)
    json.dump(metadata, open(metadata_path, "w"))
    
    hour_count += len(meta) + 1
    
    if hour_count >= hour_amount or do_sleep:
        hour_count = 0
        try:
            print("Changing to the next token")
            logging.info("Changing to the next token")
            api = get_api(next(tokens))
        except StopIteration:
            print("Sleeping...")
            logging.info("Sleeping...")
            tokens = iter(cred['tokens'])
            api = get_api(next(tokens))
            sleep(3600)
    