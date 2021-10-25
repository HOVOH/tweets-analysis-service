from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import re


def remove_urls(string):
    return re.sub("https?:\/\/.*[\r\n]*", "", string)


def remove_users(string):
    return re.sub("@\\w*", "", string)


def remove_retweets(string):
    return re.sub("^(rt|RT)\\s@\\w*:", "", string)


def stem_sentence(string, language="english"):
    tokens = [w.lower() for w in word_tokenize(string)]
    words = [w for w in tokens if not w in set(stopwords.words(language))]
    words = [w for w in words if w.isalnum()]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    return stemmed


class TextCleaner:

    def __init__(self, stem=True, remove_urls=True, remove_retweets=True, remove_users=True):
        self.stem = stem
        self.remove_urls = remove_urls
        self.remove_retweets = remove_retweets
        self.remove_users = remove_users

    def fit(self, x, y):
        return self

    def transform(self, x):
        for col in x:
            if self.remove_retweets:
                x.loc[:, col] = x.loc[:, col].apply(remove_retweets)
            if self.remove_urls:
                x.loc[:, col] = x.loc[:, col].apply(remove_urls)
            if self.remove_users:
                x.loc[:, col] = x.loc[:, col].apply(remove_users)
            if self.stem:
                x.loc[:, col] = x.loc[:, col].apply(lambda sentence: " ".join(stem_sentence(sentence)))
        return x

    def set_params(self, stem=True, remove_urls=True, remove_retweets=True, remove_users=True):
        self.stem = stem
        self.remove_urls = remove_urls
        self.remove_retweets = remove_retweets
        self.remove_users = remove_users
