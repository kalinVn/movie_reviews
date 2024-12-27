import nltk
import pandas as pd

import re
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

import config


class MovieReviewNLTK:

    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_train_count = None
        self.x_test_count = None
        self.x_train_tfidf = None
        self.x_test_tfidf = None
        csv_path = config.CSV_PATH
        self.dataset = pd.read_csv(csv_path)
        self.count_vectorized = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))
        self.tf_vectorized = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3))

    def set_train_test_data(self):
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

    def get_x_train(self):
        return self.x_train

    def get_x_train_count(self):
        return self.x_train_count

    def get_x_train_tfidf(self):
        return self.x_train_tfidf

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def get_x_test(self):
        return self.x_test

    def get_x_test_count(self):
        return self.x_test_count

    def get_x_test_tfidf(self):
        return self.x_test_tfidf



