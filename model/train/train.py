# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import sys
import os 
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..') )

# Common imports
import numpy as np
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(42)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from utils.utils import fetch_spam_data, load_email, get_email_structure, email_to_text
from NLP.TextProcessor import TextProcessor
from textblob import TextBlob
from collections import Counter
from joblib import dump
from NLP.Transformers import WordCounterToVectorTransformer, StructureTransformer
import pickle



DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("model", "datasets", "spam")

HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")

DUMP_PATH = os.path.join("model", "dump")

fetch_spam_data(SPAM_URL, HAM_URL, SPAM_PATH)

print('------------------Data fetched----------------------------')

ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]

ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

ham_emails = [email for email in ham_emails if TextBlob(email_to_text(email) or 'bonjour').detect_language()=='en']
spam_emails = [email for email in spam_emails if TextBlob(email_to_text(email) or 'bonjour').detect_language()=='en']

print('------------------Data filtered----------------------------')


structures = [get_email_structure(email) for  email in ham_emails]
structures = structures + [get_email_structure(email) for  email in spam_emails]

contents = [email_to_text(email) or '' for email in ham_emails]
contents = contents + [email_to_text(email) or '' for email in spam_emails]

subjects = [email['Subject'] for email in ham_emails]
subjects = subjects + [email['Subject'] for email in spam_emails]

data = pd.DataFrame({"subject" : subjects, "content" : contents, "structure": structures, 
                     'target' : np.array([0] * len(ham_emails) + [1] * len(spam_emails))})
data.drop_duplicates(['subject', 'content'], inplace=True)

tp = TextProcessor()
data_processed = data.copy()
data_processed.content = data_processed.content.apply(lambda x: tp.process(x, allow_stopwords=False, use_stemmer=True))
data_processed.subject = data_processed.subject.apply(lambda x : tp.process(x, allow_stopwords = True, use_stemmer=True))
data_processed['whole']  = data_processed.subject + ' ' + data_processed.content

print('------------------Data processed----------------------------')

preprocess_pipeline = ColumnTransformer([
    ("wordcount_to_vector", WordCounterToVectorTransformer(), ['whole']),
    ("structure_transformer", StructureTransformer(), ['structure']),
    #("tfidf", TfIdfTransformer(), ['whole']),
    #("lda_transformer", LdaTransformer(), ['whole']),
])
model = LogisticRegression(solver="liblinear", random_state=42)

full_pipeline = Pipeline([
    ('preprocessor', preprocess_pipeline),
    ('model', model)
])

data_processed = data_processed.reset_index(drop=True)
random_permutation = np.random.permutation(len(data_processed))
data_processed = data_processed.loc[random_permutation]
data_processed = data_processed.reset_index(drop=True)

full_pipeline.fit(data_processed.drop('target', axis=1), data_processed.target.values.ravel())

print('------------------Model trained----------------------------')

d = {'data': data,
    'data_processed' : data_processed}

with open(os.path.join(DUMP_PATH, 'dump.pkl'), 'wb') as file:
    pickle.dump(d, file)
    
dump(full_pipeline, 'model.joblib') 