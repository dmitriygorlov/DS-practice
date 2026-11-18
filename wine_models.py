"""Serializable text preprocessing and leakage-free regression pipelines."""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def lemmatize_texts(texts):
    import nltk
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    tags = {'J': 'a', 'N': 'n', 'V': 'v', 'R': 'r'}
    return [' '.join(lemmatizer.lemmatize(word, tags.get(tag[0], 'n'))
                     for word, tag in nltk.pos_tag(re.findall('[a-zA-Z]+', str(text).lower())))
            for text in texts]


def text_pipeline(model, stopwords, lemm=False):
    steps = []
    if lemm:
        steps.append(('lemmatize', FunctionTransformer(lemmatize_texts, validate=False)))
    steps.extend([('tfidf', TfidfVectorizer(stop_words=list(stopwords), ngram_range=(1, 2))),
                  ('model', model)])
    return Pipeline(steps)
