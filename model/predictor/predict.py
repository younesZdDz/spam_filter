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
from joblib import load
from NLP.Transformers import WordCounterToVectorTransformer, StructureTransformer
import pickle

