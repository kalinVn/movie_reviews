import pandas as pd
from keras_preprocessing import sequence
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

class MovieReviewKeras:
    def __init__(self):
        self.training_set = None
        self.testing_set = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_train_padded = None
        self.x_test_padded = None
        self.tokenizer = None

    def set_train_test_data(self):
        data = pd.read_csv('data/IMDB Dataset.csv')

        data.replace({
            'sentiment': {
                'positive': 1, 'negative': 0
            }
        }, inplace=True)

        self.training_set, self.testing_set =  train_test_split(data, test_size=0.2, random_state=42)

        self.tokenizer = Tokenizer(num_words=5000)

        self.tokenizer.fit_on_texts(self.training_set['review'])

        self.x_train_padded = sequence.pad_sequences(self.tokenizer.texts_to_sequences(self.training_set['review']), maxlen=200)
        self.x_test_padded = sequence.pad_sequences(self.tokenizer.texts_to_sequences(self.testing_set['review']), maxlen=200)

        self.y_train = self.training_set['sentiment']
        self.y_test = self.testing_set['sentiment']

        print("X_train vector shape = {x_train_padded_shape}".format(x_train_padded_shape=self.x_train_padded.shape))
        print("X_test vector shape = {x_test_padded_shape}".format(x_test_padded_shape=self.x_test_padded.shape))

    def tokenize_review(self, review):
        padded_sequence =  sequence.pad_sequences(self.tokenizer.texts_to_sequences(review), maxlen=200)
        return padded_sequence

    def get_x_train_padded(self):
        return self.x_train_padded

    def get_x_test_padded(self):
        return self.x_test_padded

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def get_x_train(self):
        return self.x_train

    def get_x_test(self):
        return self.x_test

