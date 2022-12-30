import json
import os
import random

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
metadata_path = "dataset/metadata.json"
img_basedir = "dataset/images"
ds_metadata = {}

pre_metadata = json.load(open(pre_metadata_path, "r"))
# metadata = json.load(open(metadata_path, "r"))
# json.dump(metadata, open(f"dataset/backup/metadata.{ datetime.now().strftime('%Y-%m-%d_%H:%M:%S') }.json", "w"))

all_paths = pre_metadata['paths']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

queries = list(pre_metadata['queries'].keys())
ds_metadata['stat'] = { k: 0  for k in queries }
ds_metadata['paths'] = { k: [] for k in queries }

with torch.no_grad():
    for i, q in enumerate(tqdm(queries)):
        image_buffer, paths_buffer = [], []
        query_buffer = [ "a photo of " + q.lower() ]
        acc_thres = 0.5
        
        paths = all_paths[q]
        while len(query_buffer) < 8:
            neg_q = "a photo of " + random.choice(queries).lower()
            if not neg_q in query_buffer:
                query_buffer.append(neg_q)
                
        for p in paths:
            image_buffer.append(preprocess(Image.open(os.path.join(img_basedir, p))).to(device))
            
        image = torch.stack(image_buffer)
        query = clip.tokenize(query_buffer).to(device)
        image_features = model.encode_image(image)
        query_features = model.encode_text(query)
        logits_per_image, logits_per_text = model(image, query)
        probs = logits_per_image.softmax(dim=-1)
        
        acc = torch.where(probs[:, 0] > acc_thres)[0].cpu().numpy().tolist()
        ds_metadata['stat'][q] = len(acc)
        ds_metadata['paths'][q] = [ paths[j] for j in acc ]
        
        tqdm.write(f"[{q}] Accepted {len(acc)} / {len(paths)} images")
        
        image_buffer, query_buffer, paths_buffer = [], [], []
    
json.dump(ds_metadata, open(metadata_path, "w"))
    