#!/usr/bin/python

import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import re, string
import sys
import os
import math
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def predict_proba(plot):

  model = joblib.load('model_lg.pkl')
  # vect = joblib.load('vect.pkl')

  # Transform x
  plot=[plot]
  #X = vect.transform(['plot'])

  re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡!.,§£₤‘’])')
  wordnet_lemmatizer = WordNetLemmatizer()
  def split_into_lemmas(text):
    text = text.lower()
    words = text.split()
    return [wordnet_lemmatizer.lemmatize(word, 's') for word in words]
  def tokenize(s): return re_tok.sub(r' \1 ', s).split()
  vect = TfidfVectorizer(analyzer=split_into_lemmas, ngram_range=(1,4), tokenizer=tokenize, max_features=None,
               min_df=1, max_df=1.9, strip_accents='unicode', use_idf=0.1, stop_words = 'english',
               smooth_idf=0.1, sublinear_tf=True)#, vocabulary = voc)
  X = vect.fit_transform(plot)
  
  y_pred_genres = model.predict_proba(X)

  p1 = "Based on the plot, gender of the movie is " + str(y_pred_genres) 

  return p1

