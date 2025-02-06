from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# Improved Bag of Words Vectorizer
class BowVectors(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bow_vectorizer = CountVectorizer()

    def fit(self, df):
        self.bow_vectorizer.fit(df)
        return self

    def transform(self, df):
        features = self.bow_vectorizer.transform(df)
        return features

# Improved TF-IDF Vectorizer
class TfidfVectors():
    def __init__(self, lowercase=True, ngram_range=(1, 1)):
        self.tfidf_vectorizer = TfidfVectorizer(lowercase=lowercase, ngram_range=ngram_range)

    def fit_transform(self, df):
        features = self.tfidf_vectorizer.fit_transform(df)
        return features

    def transform(self, df):
        features = self.tfidf_vectorizer.transform(df)
        return features

# Generic Vectorizer Class
class Vectorizer:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit_transform(self, text_array):
        if isinstance(text_array, list):
            return self.vectorizer.fit_transform(text_array)
        elif isinstance(text_array, str):
            return self.vectorizer.fit_transform([text_array])
        else:
            raise ValueError("Input should be a list or string")

    def transform(self, text_array):
        if isinstance(text_array, list):
            return self.vectorizer.transform(text_array)
        elif isinstance(text_array, str):
            return self.vectorizer.transform([text_array])
        else:
            raise ValueError("Input should be a list or string")
