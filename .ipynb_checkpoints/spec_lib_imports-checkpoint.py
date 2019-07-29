def my_lib_imports():
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
    from sklearn.metrics import classification_report
    return