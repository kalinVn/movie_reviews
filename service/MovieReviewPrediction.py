
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

import config
from factory.Classifier import Classifier
from factory.PreprocessFactory import PreprocessFactory
from sklearn.metrics import accuracy_score
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
        preprocess_factory = PreprocessFactory(config.MOVIE_REVIEW_PREDICTION_NLTK)
        self.preprocess = preprocess_factory.get_obj()
        self.classifier_factory = Classifier()
        self.model = self.classifier_factory.get_model()

        self.label_b = LabelBinarizer()

    def get_dataset(self):
        return self.dataset

    def get_preprocess(self):
        return self.preprocess

    def build(self):
        # print(self.preprocess.get_x_train_count(), self.preprocess.get_y_train)
        # return
        self.logistic_model_bow = self.model.fit(self.preprocess.get_x_train_count(), self.preprocess.get_y_train())
        self.logistic_model_tfidf = self.model.fit(self.preprocess.get_x_train_tfidf(), self.preprocess.get_y_train())

    def test_accuracy_score(self):
        self.lr_bow_prediction = self.logistic_model_bow.predict(self.preprocess.get_x_test_count())
        self.lr_tfidf_prediction = self.logistic_model_tfidf.predict(self.preprocess.get_x_test_tfidf())

        bow_training_data_accuracy = accuracy_score(self.preprocess.get_y_test(),  self.lr_bow_prediction)
        print("Accuracy score of bow training data: ", bow_training_data_accuracy)

        tfidf_training_data_accuracy = accuracy_score(self.preprocess.get_y_test(), self.lr_tfidf_prediction)
        print("Accuracy score of tfidf training data: ", tfidf_training_data_accuracy)



