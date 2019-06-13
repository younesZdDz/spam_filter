# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import sys
import os 
sys.path.append(os.path.join(os.getcwd(), 'model') )

import traceback
import time
start_time = time.time()

# Common imports
import numpy as np
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(42)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from utils.utils import get_email_structure, email_to_text
from NLP.TextProcessor import TextProcessor
from textblob import TextBlob
from collections import Counter
from joblib import load
from NLP.Transformers import WordCounterToVectorTransformer, StructureTransformer
import pickle

import email
import email.policy

import nltk
nltk.download('wordnet')

from flask import request
from flask import Flask
from flask import render_template

import json

DUMP_PATH = os.path.join("model", "dump")

model = load('model/dump/model.joblib') 


# init flask server
app = Flask(__name__, static_url_path='', template_folder='view')

# Routes
print("--- %s seconds ---" % (time.time() - start_time))

@app.route('/')
def index():
  return render_template('index.html')


@app.route("/api/check_mail", methods = ['POST'])
def check_mail():
    try:    
        f = request.files['file']
        f.save('/tmp/email_sample.eml')
        with open('/tmp/email_sample.eml', "rb") as f:
          email_sample= email.parser.BytesParser(policy=email.policy.default).parse(f)

        structure = get_email_structure(email_sample)
        subject = email_sample['Subject']
        content = email_to_text(email_sample) or '' 
        data = pd.DataFrame({"subject" : [subject], "content" : [content], "structure": [structure], })
        tp = TextProcessor()
        data.content = data.content.apply(lambda x: tp.process(x, allow_stopwords=False, use_stemmer=True))
        data.subject = data.subject.apply(lambda x : tp.process(x, allow_stopwords = True, use_stemmer=True))
        data['whole']  = data.subject + ' ' + data.content  
        pred = model.predict(data)
        return json.dumps([{'target' : pred[0]}], default=str)

    except Exception as e:
        traceback.print_exc()
        return json.dumps([], default=str)


@app.route("/api/check_mail_form", methods = ['POST'])
def check_mail_form():
    try:    
        data = request.get_json()
        print(data)
        structure = 'text/plain'
        subject = data['email_subject']
        content = data['email_content']
        data = pd.DataFrame({"subject" : [subject], "content" : [content], "structure": [structure], })
        tp = TextProcessor()
        data.content = data.content.apply(lambda x: tp.process(x, allow_stopwords=False, use_stemmer=True))
        data.subject = data.subject.apply(lambda x : tp.process(x, allow_stopwords = True, use_stemmer=True))
        data['whole']  = data.subject + ' ' + data.content  
        pred = model.predict(data)
        return json.dumps([{'target' : pred[0]}], default=str)

    except Exception as e:
        traceback.print_exc()
        return json.dumps([], default=str)

if __name__ == '__main__':
  app.run(debug=False, host='0.0.0.0', port = 8000)