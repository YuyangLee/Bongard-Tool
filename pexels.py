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


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=f'logs/pexels-{ str(datetime.now().timestamp()) }.log', format=LOG_FORMAT)

cred_path = "credentials/credentials.yaml"
query_path = "data/queries.json"

metadata_path = "dataset/metadata.json"
dataset_basedir = "dataset/images"

cred = get_yaml_data(cred_path)['pexels']

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

token = next(tokens)

hour_count = 0

for query in tqdm(queries[28:]):
    meta, do_sleep = download_pexels(query, token, dataset_basedir, n_each_query)
    
    metadata['metadata'] += meta
    metadata['queries'][query] = len(meta) if query not in metadata['queries'] else metadata['queries'][query] + len(meta)
    json.dump(metadata, open(metadata_path, "w"))
    
    hour_count += len(meta) + 1
    
    if hour_count >= hour_amount or do_sleep:
        hour_count = 0
        try:
            print("Changing to the next token")
            logging.info("Changing to the next token")
            token = next(tokens)
        except StopIteration:
            print("Sleeping...")
            logging.info("Sleeping...")
            tokens = iter(cred['tokens'])
            token = next(tokens)
            sleep(3600)
    