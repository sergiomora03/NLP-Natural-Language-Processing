#!/usr/bin/python

import pandas as pd
#from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import joblib
import re, string
import sys
import os
import math
import nltk
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def predict_proba(plot):

  model = joblib.load('model.pkl')

  cols = ['p_Action','p_Adventure','p_Animation','p_Biography','p_Comedy','p_Crime','p_Documentary','p_Drama','p_Family','p_Fantasy','p_Film-Noir','p_History','p_Horror','p_Music','p_Musical','p_Mystery','p_News','p_Romance','p_Sci-Fi','p_Short','p_Sport','p_Thriller','p_War','p_Western']
 
  with open("vocabulary.txt","rb") as fp:
    voc_ = pickle.load(fp)
 
  re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡!.,§£₤‘’])')
  wordnet_lemmatizer = WordNetLemmatizer()
  def split_into_lemmas(text):
    text = text.lower()
    words = text.split()
    return [wordnet_lemmatizer.lemmatize(word,'s') for word in words]
  def tokenize(s): return re_tok.sub(r' \1 ', s).split()
  vect = TfidfVectorizer(analyzer=split_into_lemmas, ngram_range=(1,4), max_features=None, tokenizer=tokenize,
               min_df=1, max_df=1.9, strip_accents='unicode', use_idf=0.001, stop_words = 'english',
               smooth_idf=0.1, sublinear_tf=True, vocabulary = voc_)
 
  X = vect.fit_transform([plot])#vect.transform([plot])
 
  res = pd.DataFrame(model.predict_proba(X), index=range(1), columns=cols)
  p1 = res[res.iloc[0:,:] >= 0.5].dropna(axis = 1).to_string()

  #p1 = "Based on the plot, gender of the movie is " + str(X) 

  return p1

