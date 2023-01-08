import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

import re, string, unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from textblob import TextBlob
from textblob import Word

import config
from factory.Classifier import Classifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from bs4 import BeautifulSoup


class MovieReviewPrediction:

    def __init__(self):

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.logistic_model_bow = None
        self.logistic_model_tfidf = None
        self.lr_bow_prediction = None
        self.lr_tfidf_prediction = None
        self.x_train_count = None
        self.x_test_count = None
        self.x_train_tfidf = None
        self.x_test_tfidf = None

        csv_path = config.CSV_PATH
        self.dataset = pd.read_csv(csv_path)

        self.classifier_factory = Classifier()
        self.model = self.classifier_factory.get_model()
        self.count_vectorized = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))
        self.tf_vectorized = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3))

        self.label_b = LabelBinarizer()

    def get_dataset(self):
        return self.dataset

    def preprocess(self):
        nltk.download('stopwords')
        self.dataset['review'] = self.dataset['review'].apply(self.noise_remove_vals)
        self.dataset['review'] = self.dataset['review'].apply(self.stemmer)
        self.dataset['review'] = self.dataset['review'].apply(self.removing_stopwords)

        self.x_train = self.dataset.review[:config.TRAINING_DATA_SIZE]
        self.x_test = self.dataset.review[config.TRAINING_DATA_SIZE:]
        self.x_train_count = self.count_vectorized.fit_transform(self.x_train)
        self.x_test_count = self.count_vectorized.transform(self.x_test)

        self.x_train_tfidf = self.tf_vectorized.fit_transform(self.x_train)
        self.x_test_tfidf = self.tf_vectorized.transform(self.x_test)

        self.label_b.fit_transform(self.dataset['sentiment'])
        self.y_train = self.dataset.sentiment[:config.TRAINING_DATA_SIZE]
        self.y_test = self.dataset.sentiment[config.TRAINING_DATA_SIZE:]


    def noise_remove_vals(self, text):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        text = re.sub('\[[^]]*\]', '', text)

        return text

    def stemmer(self, text):
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])

        return text

    def removing_stopwords(self, text):
        stop_words = nltk.corpus.stopwords.words('english')
        tokenizers = ToktokTokenizer()

        tokens = tokenizers.tokenize(text)
        tokens = [i.strip() for i in tokens]

        fill_tokens = [i for i in tokens if i.lower() not in stop_words]

        filtered_text = ' '.join(fill_tokens)

        return filtered_text

    def build(self):

        self.logistic_model_bow = self.model.fit(self.x_train_count, self.y_train)
        self.logistic_model_tfidf = self.model.fit(self.x_train_tfidf, self.y_train)

    def test_accuracy_score(self):
        self.lr_bow_prediction = self.logistic_model_bow.predict(self.x_test_count)
        self.lr_tfidf_prediction = self.logistic_model_tfidf.predict(self.x_test_tfidf)

        bow_training_data_accuracy = accuracy_score(self.y_test,  self.lr_bow_prediction)
        print("Accuracy score of bow training data: ", bow_training_data_accuracy)

        tfidf_training_data_accuracy = accuracy_score(self.y_test, self.lr_tfidf_prediction)
        print("Accuracy score of tfidf training data: ", tfidf_training_data_accuracy)



