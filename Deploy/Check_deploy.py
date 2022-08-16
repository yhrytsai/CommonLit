import requests
import json
import pandas as pd
import random

API_url = 'http://127.0.0.1:5000/'
path_to_sample_with_features = 'commonlitreadabilityprize/test.csv'

data = pd.read_csv(path_to_sample_with_features, usecols=['excerpt'])

for i in random.sample(range(0, len(data)), 5):
    s = data.iloc[i]
    result = requests.post(url=API_url, json=json.loads(s.to_json())).json()
    print(result)
