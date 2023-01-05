import json
import os

data_se = json.load(open("dataset/final_bing/paths.json", 'r'))
data_stocks = json.load(open("dataset/final_stocks/paths.json", 'r'))

data = { k: [] for k in data_se.keys() }

for k, v in data_se.items():
    data[k] += [os.path.join(k, p) for p in v]
    
for k, v in data_stocks.items():
    data[k] += [os.path.join(k, p) for p in v]
    
json.dump(data, open("dataset/all_paths.json", 'w'))
