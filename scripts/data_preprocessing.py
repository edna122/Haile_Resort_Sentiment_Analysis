import pandas as pd
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer  # Or CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Or your chosen model

# 1. Stop Words
stop_words = set(stopwords.words('english'))
hotel_stopwords = {
    "hotel", "room", "staff", "stay", "guest", "service", "location", "breakfast",
    "restaurant", "pool", "beach", "resort", "experience", "time", "day", "night",
    "place", "like", "also", "get", "go", "went", "came", "back", "really", "very",
    "much", "little", "bit", "quite", "even", "though", "however", "but",
    "so", "just", "one", "two", "three",  # Numbers might not be helpful
    "us", "we", "our", "my", "your", "their" # Pronouns, often not helpful for sentiment
}

stop_words.update(hotel_stopwords)

stop_words.discard("not")
stop_words.discard("good")
stop_words.discard("bad")
stop_words.discard("excellent")
stop_words.discard("poor")
stop_words.discard("great")


# 2. Contraction Expansion
def contraction_expansion(content):
    content = re.sub(r"won\'t", "would not", content)
    content = re.sub(r"can\'t", "can not", content)
    content = re.sub(r"don\'t", "do not", content)
    content = re.sub(r"shouldn\'t", "should not", content)
    content = re.sub(r"needn\'t", "need not", content)
    content = re.sub(r"hasn\'t", "has not", content)
    content = re.sub(r"haven\'t", "have not", content)
    content = re.sub(r"weren\'t", "were not", content)
    content = re.sub(r"mightn\'t", "might not", content)
    content = re.sub(r"didn\'t", "did not", content)
    content = re.sub(r"n\'t", " not", content)
    content = re.sub(r"i\'m", "i am", content)
    content = re.sub(r"it\'s", "it is", content)
    content = re.sub(r"\'s", " is", content)
    content = re.sub(r"\'ve", " have", content)
    content = re.sub(r"\'re", " are", content)
    content = re.sub(r"\'d", " would", content)
    return content


# 3. Other Text Cleaning Functions
def remove_special_character(content):
    return re.sub('\W+', ' ', content)

def remove_url(content):
    return re.sub(r'http\S+', '', content)

def remove_stopwords(content):
    clean_data = []
    for i in content.split():
        if i.strip().lower() not in stop_words and i.strip().lower().isalpha():
            clean_data.append(i.strip().lower())
    return " ".join(clean_data)


def data_cleaning(content):
    content = contraction_expansion(content)
    content = remove_special_character(content)
    content = remove_url(content)
    content = remove_stopwords(content)
    return content


# 4. Data Cleaning Transformer
class DataCleaning(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        return X.apply(data_cleaning)  # Optimized apply

# 5. LemmaTokenizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()
    def __call__(self, reviews):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]

# 6. Vectorization (Choose one)
vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer())  # TF-IDF with lemmatization

