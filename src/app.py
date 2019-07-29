import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
import pickle

from sklearn.feature_extraction.text \
    import CountVectorizer, TfidfTransformer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from os import path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

import re
import pickle

from sklearn.naive_bayes import MultinomialNB

cv = CountVectorizer(lowercase=True, tokenizer=None, strip_accents= "ascii",\
                     stop_words="english", analyzer='word', max_df=1.0,\
                     min_df=2,ngram_range=(1,2), max_features=3000)
tfidf = TfidfTransformer(use_idf=True)
wordnet = WordNetLemmatizer()

def lsts_words_df_str(lsts):
    """Purpose:  Takes a list of lists where each list contains words \
    that have been cleaned,tokenized and lemmatized. These words are rejoined \
    into documents and inserted into a new dataframe

    Input: lists of tokenized documents
    Output: cleaned dataframe with a column of text"""

    new_docs =[]
    for element in lsts:
        doc = " ".join(element)
        new_docs.append(doc)
    new_series_docs = pd.Series(new_docs)
    cleaned_df = pd.DataFrame(new_series_docs,columns = ["text"])
    return cleaned_df


def cleaned_dframe(df, col_name = None):
    """Purpose: Take in a text based Dataframe and return a cleaned text
    dataframe by using regex, lowercasing, stripping stop words and lemmatizing

    Input: Dataframe with only text column
    Output: Dataframe with only cleaned text column"""

    df[col_name] = df[col_name].str.replace(r'([^a-zA-Z\s]+?)'," ")
    df[col_name] = df[col_name].apply(lambda x : x.lower())
    docs_tokenized = [word_tokenize(content) for content in df[col_name].values]
    stop = set(stopwords.words('english'))
    docs_stop = [[word for word in words if word not in stop] for words in\
                 docs_tokenized]
    docs_wordnet = [[wordnet.lemmatize(word) for word in words] for words in\
                    docs_stop]
    cleaned_df = lsts_words_df_str(docs_wordnet)
    return cleaned_df


def text2num(cleaned_df, col= None, train=True, cv=None, tfidf=None):
    """Purpose: receive a df and a column for text and turn text into\
    CountVectorized Data and tfidf data

    Input: DataFrame with string column to be numerically vectorized
    Output: A doc word count matrix and a tfidf matrix as well"""

    str_data = cleaned_df[col].values
    if train == True:
        X_counts = cv.fit_transform(str_data)
        X_counts_tfidf_arr = tfidf.fit_transform(X_counts).toarray()
    else:
        X_counts = cv.transform(str_data)
        X_counts_tfidf_arr = tfidf.transform(X_counts).toarray()
    return X_counts, X_counts_tfidf_arr


if __name__=="__main__":
    review = ["I hated this movie"]
    y_test = 1
    df_test = pd.DataFrame(review,columns = ["Review"])
    wordnet = pickle.load(open("wordnet.pkl", 'rb'))
    df_test_clean = cleaned_dframe(df_test.copy(),"Review")
    cv = pickle.load(open("cv_model.pkl", 'rb'))
    tfidf = pickle.load(open("tfidf_model.pkl", 'rb'))
    X_counts_test, X_counts_tfidf_arr_test = text2num(df_test_clean.copy(),\
                                                   "text",False,cv,tfidf)
    print(X_counts_test.shape, X_counts_tfidf_arr_test.shape)
    nb_model = pickle.load(open("nb_model.pkl", 'rb'))
    y_pred = nb_model.predict(X_counts_tfidf_arr_test)
    print(y_pred)
