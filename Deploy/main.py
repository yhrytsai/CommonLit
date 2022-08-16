'''
Expected input - json with next field:
{   excerpt: text# - text to predict reading ease of
}

Output - json with next field:
{
    target: text # reading ease
}
'''

################################################### Upload packages ####################################################
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import SVR
import pandas as pd
import flask
import json
import joblib

################################################# Load model functions ##################################################
def count_capital_char(text):
    count = 0
    for i in text:
        if i.isupper():
            count += 1
    return count

def features_creation(data, col):
    data['character_number'] = data.apply(lambda x: len(x[col]), axis=1)

    data['words_number'] = data.apply(lambda x: len(x[col].split()), axis=1)

    data['capital_character_number'] = data.apply(lambda x: count_capital_char(x[col]), axis=1)

    data['capital_words_number'] = data.apply(lambda x: sum(map(str.isupper, x[col].split())), axis=1)

    data['punctuation_number'] = data.apply(lambda x: sum(map(str.isupper, x[col].split())), axis=1)

    data['sentences_number'] = data.apply(lambda x: x[col].count("."), axis=1)

    data['unique_words_number'] = data.apply(lambda x: len(set(x[col])), axis=1)

    data['wordlength_avg'] = data['character_number'] / data['words_number']

    data['sentlength_avg'] = data['words_number'] / data['sentences_number']

    data['unique_vs_words'] = data['unique_words_number'] / data['words_number']

    return data

path_to_pipeline = 'src/model_pipeline.pkl'
mod = joblib.load(path_to_pipeline)

################################################# Deployment function ##################################################
app = flask.Flask(__name__)

# defining a route for only post requests
@app.route('/', methods=['POST'])

def index():
    data = flask.request.get_json(force=True)
    data = pd.DataFrame([data])
    data = features_creation(data, 'excerpt')
    data['target'] = mod.predict(data)
    rez = json.dumps({'target': str(data['target'].values[0].round(3))})
    return rez


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)