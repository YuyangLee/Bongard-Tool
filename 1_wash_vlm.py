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
from utils.utils_clip import export_transform

pre_metadata_path = "dataset/pre_metadata.json"
metadata_path = "dataset/metadata.json"
img_basedir = "dataset/images"
export_basedir = "dataset/functools"
ds_metadata = {}
unified_resolution = 1024

n_acc, n_all = 0, 0

pre_metadata = json.load(open(pre_metadata_path, "r"))
# metadata = json.load(open(metadata_path, "r"))
# json.dump(metadata, open(f"dataset/backup/metadata.{ datetime.now().strftime('%Y-%m-%d_%H:%M:%S') }.json", "w"))

all_paths = pre_metadata['paths']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
export_proc = export_transform(unified_resolution)

queries = list(pre_metadata['queries'].keys())
ds_metadata['stat'] = { k: 0  for k in queries }
ds_metadata['paths'] = { k: {} for k in queries }

with torch.no_grad():
    for i, q in enumerate(tqdm(queries)):
        image_buffer, paths_buffer, export_buffer = [], [], []
        query_buffer = [ "a photo of " + q.lower() ]
        acc_thres = 0.5
        
        paths = all_paths[q]
        ids = list(paths.keys())
        n_all += len(paths)
        
        while len(query_buffer) < 8:
            neg_q = "a photo of " + random.choice(queries).lower()
            if not neg_q in query_buffer:
                query_buffer.append(neg_q)
                
        for p in paths.values():
            image = Image.open(os.path.join(img_basedir, p))
            image_buffer.append(preprocess(image).to(device))
            export_buffer.append(export_proc(image).to(device))
            
        image = torch.stack(image_buffer)
        query = clip.tokenize(query_buffer).to(device)
        image_features = model.encode_image(image)
        query_features = model.encode_text(query)
        logits_per_image, logits_per_text = model(image, query)
        probs = logits_per_image.softmax(dim=-1)
        
        acc = torch.where(probs[:, 0] > acc_thres)[0].cpu().numpy().tolist()
        n_acc += len(acc)
        ds_metadata['stat'][q] = len(acc)
        ds_metadata['paths'][q] = { ids[j]: paths[ids[j]] for j in acc }
        
        tqdm.write(f"[{q}] Accepted {len(acc)} / {len(paths)} images")
        
        os.makedirs(os.path.join(export_basedir, q), exist_ok=True)
        for j, id in enumerate(paths.keys()):
            filename = id + ".jpg"
            if not j in acc:
                filename = "rej_" + filename
            torchvision.utils.save_image(export_buffer[j], os.path.join(export_basedir, q, filename))
        
        image_buffer, query_buffer, paths_buffer = [], [], []
    
json.dump(ds_metadata, open(metadata_path, "w"))

print(f"Totally accepted {n_acc} / {n_all} images")
    