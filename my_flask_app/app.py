from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from os import path
import joblib
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
wordnet = WordNetLemmatizer()
import re
app = Flask(__name__)
pos=[]
neg=[]
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
#    wordnet = open('data/wordnet.pkl','rb')
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
@app.route('/')
def home():
   global pos
   global neg
   return render_template('home.html', pos=pos, neg=neg)
@app.route('/predict',methods=['POST'])
def predict():
   global pos
   global neg
   nb_model = joblib.load('data/nb_model.pkl')
#    wordnet = open('data/wordnet.pkl','rb')
   tfidf = joblib.load('data/tfidf_model.pkl')
   cv = joblib.load('data/cv_model.pkl')
   if request.method == 'POST':
        message = request.form['message']
        data = [message]
        df_data = pd.DataFrame(data,columns=["Review"])
        df_data_clean = cleaned_dframe(df_data.copy(),"Review")
        X_counts_data, X_counts_tfidf_arr_data = text2num(df_data_clean.copy(),
                                                          "text",False,cv,tfidf)
        my_prediction = nb_model.predict(X_counts_tfidf_arr_data)
        if my_prediction == True:
#            prob=int(100*np.around(nb_model.predict_proba(X_counts_tfidf_arr_data)[0,1],decimals=2))
            pos.insert(0, message)
            if len(pos) > 10:
                pos=pos[0:10]
        else:
#            prob=int(100*np.around(nb_model.predict_proba(X_counts_tfidf_arr_data)[0,0],decimals=2))
            neg.insert(0, message)
            if len(neg) > 10:
                neg=neg[0:10]
        return render_template('result.html',prediction = my_prediction)
#        return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
