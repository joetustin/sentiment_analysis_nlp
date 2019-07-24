import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

import re

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

df=pd.read_csv("data/rotten_tomatoes_reviews.csv")

cv = CountVectorizer(lowercase=True, tokenizer=None, strip_accents= "ascii", stop_words="english",
                             analyzer='word', max_df=1.0, min_df=2,ngram_range=(1,1),
                             max_features=30000)
tfidf = TfidfTransformer(use_idf=True)

def cleaned_dframe(df, col_name = None):
    """Purpose: Take in a text based Dataframe and return a cleaned text
    dataframe by using regex, lowercasing, stripping stop words and lemmatizing

    Input: Dataframe with only text column
    Output: Dataframe with only cleaned text column"""

    #using regexp notation to get rid of numbers in reviews
    df[col_name] = df[col_name].str.replace(r'([^a-zA-Z\s]+?)'," ")

    # 1. Create a set of documents.
    df[col_name] = df[col_name].apply(lambda x : x.lower())

    # 2. Create a set of tokenized documents.
    docs_tokenized = [word_tokenize(content) for content in df[col_name].values]


    # 3. Strip out stop words from each tokenized document.

    stop = set(stopwords.words('english'))
#    new_stopwords = set(["film","movie","like","feel","time","little","adject", "adds",
#                        "bestloved","agonizingly","bantamweight"])
#    stop.update(new_stopwords)
    docs_stop = [[word for word in words if word not in stop] for words in docs_tokenized]

    # Stemming / Lemmatization

    # 1. Stem using lemmatizer
    wordnet = WordNetLemmatizer()
    docs_wordnet = [[wordnet.lemmatize(word) for word in words] for words in docs_stop]

    new_element =[]
    for element in docs_wordnet:
        test = " ".join(element)
        new_element.append(test)
    new_series = pd.Series(new_element)
    col = "text"
    cleaned_df = pd.DataFrame(new_series,columns = [col])
    return cleaned_df

def text2num(cleaned_df, col= None, train=True, cv=None, tfidf=None):
    """Purpose: receive a df and a column for text and turn text into CountVectorized Data and tfidf
    data

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
    df_train = df[:200000]
    df_test = df[200000:250000]
    y_train = df_train.Freshness
    y_test = df_test.Freshness
    df_train_clean = cleaned_dframe(df_train.copy(),"Review")
    df_test_clean = cleaned_dframe(df_test.copy(),"Review")
    X_counts_train, X_counts_tfidf_arr_train = text2num(df_train_clean.copy(),"text",True,cv,tfidf)
    X_counts_test, X_counts_tfidf_arr_test = text2num(df_test_clean.copy(),"text",False,cv,tfidf)
    nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    nb_model.fit(X_counts_tfidf_arr_train, y_train)
    print(nb_model.score(X_counts_tfidf_arr_train,y_train), nb_model.score(X_counts_tfidf_arr_test,y_test))
