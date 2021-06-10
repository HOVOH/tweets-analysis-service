import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
twitter_words = ["rt"]

def get_words(string, language="english"):
    string = re.sub("https?:\/\/.*[\r\n]*", "", string)
    tokens = [w.lower() for w in word_tokenize(string)]
    words = [word for word in tokens if word.isalpha()]
    words = [w for w in words if not w in set(stopwords.words(language))]
    words = [w for w in words if not w in set(twitter_words)]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    return stemmed
