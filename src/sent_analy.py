# all the basic libraries which are normally imported
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

# Some special libraries which had to be imported specifically for this project
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

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report

# make istances of models and tools to be used later in program
cv = CountVectorizer(lowercase=True, tokenizer=None, strip_accents= "ascii",\
                     stop_words="english", analyzer='word', max_df=1.0,\
                     min_df=2,ngram_range=(1,2), max_features=3000)
tfidf = TfidfTransformer(use_idf=True)
wordnet = WordNetLemmatizer()
nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)


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


def get_common_words(df_clean):
    """Purpose:  Given a cleaned dataframe create a new data frame\
    with just the most common words.  Workhorse function to create feature\
    importance list
    
    Input: a cleaned dataframe with a column labebeled 'text'
    Oupput: A new dataframe with just the most common words and frequency"""
    
    X_train_counts = cv.transform(df_clean["text"].values)
    word_freq = dict(zip(cv.get_feature_names(),\
                             np.asarray(X_train_counts.sum(axis=0)).ravel()))
    word_counter = Counter(word_freq)
    word_counter_df = pd.DataFrame(word_counter.most_common(20), \
                                   columns = ['word', 'freq'])
    return word_counter_df


def make_wordcloud(df):
    """Purpose: Make a word cloud from a cleaned text dataframe
    
    Input:  Cleaned dataframe of text strings
    Ouput:  A plotted wordcloud indicating the most frequent words"""
    
    text= " ".join(review for review in df["text"].values)
    stopwords = set(STOPWORDS)
    wordcloud =WordCloud(width=1600, height=800,\
                        stopwords=stopwords,background_color="white").generate(text)
    fig = plt.figure(figsize = (9,6),facecolor="w")
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title("Word Cloud for Reviews", fontsize=20)
    plt.savefig("images/wordcloud.png",facecolor="w")
    plt.show()
    return


def get_fn_fp(df_test,y_test,y_pred):
    """ Purpose: To be able to examine some of the false positive and \
    false negative reviews
    
    Input: dataframe of test set, actual target, and predictd target
    Output: A false negative dataframe and a false positive dataframe"""
    
    false_neg = df_test[y_pred < y_test]
    false_pos = df_test[y_pred > y_test]
    return false_neg, false_pos

if __name__=="__main__":
    df=pd.read_csv("data/rotten_tomatoes_reviews.csv")
    df_train = df[:20000]
    df_test = df[20000:25000]
    y_train = df_train.Freshness
    y_test = df_test.Freshness
    df_train_clean = cleaned_dframe(df_train.copy(),"Review")
    df_test_clean = cleaned_dframe(df_test.copy(),"Review")
    X_counts_train, X_counts_tfidf_arr_train = text2num(df_train_clean.copy(),\
                                                        "text",True,cv,tfidf)
    X_counts_test, X_counts_tfidf_arr_test = text2num(df_test_clean.copy(),\
                                                      "text",False,cv,tfidf)
    nb_model.fit(X_counts_tfidf_arr_train, y_train)
    print(nb_model.score(X_counts_tfidf_arr_train,y_train),\
          nb_model.score(X_counts_tfidf_arr_test,y_test))

