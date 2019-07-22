import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

import re

df=pd.read_csv("data/rotten_tomatoes_reviews.csv")
#df_trial = df[:10000]
wordnet = WordNetLemmatizer()

#corpus = [' '.join(df['Review']).lower() for row in df]
df_corpus = df["Review"].str.replace(r'([^a-zA-Z\s]+?)'," ")
df_corpus =df_corpus.str.lower()
docs_tokenized = [word_tokenize(content) for content in df_corpus]
stop = set(stopwords.words('english'))
docs_stop = [[word for word in words if word not in stop] for words in docs_tokenized]
docs_wordnet = [[wordnet.lemmatize(word) for word in words] for words in docs_stop]

new_element =[]
for element in docs_wordnet:
    test = " ".join(element)
    new_element.append(test)
new_series = pd.Series(new_element)
col = "text"
new_df = pd.DataFrame(new_series,columns = [col])

cv = CountVectorizer(lowercase=True, tokenizer=None, strip_accents= "ascii", stop_words="english",
                             analyzer='word', max_df=1.0, min_df=2,ngram_range=(1,4),
                             max_features=None)
X_train_counts_final = cv.fit_transform(new_df.text.values)
X_train_counts_final_arr = X_train_counts_final.toarray()

if __name__=="__main__":
    print(X_train_counts_final_arr.shape)
    print(df_trial.head())
