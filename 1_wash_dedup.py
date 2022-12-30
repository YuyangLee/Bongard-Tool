import json
import os

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import torch
import torchvision
import clip
from PIL import Image
import yaml
from tqdm import tqdm

pre_metadata_path = "dataset/pre_metadata.json"
img_basedir = "dataset/images"

metadata = json.load(open(pre_metadata_path, "r"))
# json.dump(metadata, open(f"dataset/backup/pre_metadata.{ datetime.now().strftime('%Y-%m-%d_%H:%M:%S') }.json", "w"))

stat = metadata['queries']
data = metadata['metadata']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

actual_stat = { k: 0 for k in stat.keys() }
paths = { k: [] for k in stat.keys() }

checked, error_idxs = [], []

for i, v in enumerate(tqdm(data)):
    hash = f"{v['query']}-{v['source']}-{v['pid']}"
    if hash in checked:
        error_idxs.append(i)
        continue
    file = os.path.join(v['query'], f"{v['source'].lower()}-{v['pid']}.")
    ext = "jpg"
    
    if v['source'] == 'Pexels':
        ext = v['extension'] if not v['extension'] == "jpeg" else "jpg"
        
    file = file + ext
    file = file.replace(' ', '-')
    
    if os.path.exists(os.path.join(img_basedir, file)):
        checked.append(hash)
    else:
        error_idxs.append(i)
        continue
        
    paths[v['query']].append(file)
    actual_stat[v['query']] += 1
    
error_idxs = list(set(error_idxs))
error_idxs.sort()

print(f"Removing {len(error_idxs)} duplicates or missings")
for rm_i in error_idxs[::-1]:
    del metadata['metadata'][rm_i]
    
print(f"Actual amount of image: { len(metadata['metadata']) }")

metadata['paths'] = paths
metadata['queries'] = actual_stat

json.dump(metadata, open(pre_metadata_path, "w"))
