import json
import logging
import os
import sqlite3
from datetime import datetime

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

metadata = json.load(open(metadata_path, "r"))

queries = json.load(open(query_path, "r"))
n_each_query = 20

api_stat = {}

dataset_stat = {}

token = next(tokens)

for query in tqdm(queries):
    meta = download_pexels(query, token, dataset_basedir, n_each_query)
    metadata['metadata'] += meta
    json.dump(metadata, open(metadata_path, "w"))
    
    