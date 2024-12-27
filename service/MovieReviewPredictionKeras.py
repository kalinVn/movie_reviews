from factory.PreprocessFactory import PreprocessFactory
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
import os

import config

class MovieReviewPredictionKeras:

    def __init__(self):
        preprocess_factory = PreprocessFactory(config.MOVIE_REVIEW_PREDICTION_KERAS)
        self.preprocess = preprocess_factory.get_obj()
        self.scores_sgd_model = None
        self.rms_prop_model = None
        self.adam_model = None

    def get_preprocess(self):
        return self.preprocess

    def build_model(self):
        self.adam_model = Sequential()
        self.adam_model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
        self.adam_model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
        self.adam_model.add(Dense(units=1, activation="sigmoid"))

    def fit(self):
        self.adam_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.fit_model_by_name('adam_model')

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        return setattr(self, name, value)

    def fit_model_by_name(self, model_name):
        x_train_padded = self.preprocess.get_x_train_padded()
        x_test_padded = self.preprocess.get_x_test_padded()
        y_train = self.preprocess.get_y_train()
        y_test = self.preprocess.get_y_test()

        if (os.path.exists('store/models/' + model_name)):
            self[model_name] = tf.keras.models.load_model("store/models/" + model_name)
        else:
            self[model_name].fit(x_train_padded, y_train, batch_size=128, epochs=5,
                                                          validation_data=(x_test_padded, y_test))

            path = 'store/models/{model_name}'.format(model_name=model_name)
            self[model_name].save(path)

    def evaluate(self):
        x_test = self.get_preprocess().get_x_test_padded()
        y_test = self.get_preprocess().get_y_test()

        loss, accuracy = self.adam_model.evaluate(x_test, y_test)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")

    def get_scores_sgd_model(self):
        return self.scores_sgd_model

    def get_rms_prop_model(self):
        return self.rms_prop_model


    def predict_by_review(self, review):
        padded_sequence = self.preprocess.tokenize_review(review)
        prediction = self.adam_model.predict(padded_sequence)
        print("Prediction ---------------------->")

        sentiment = "Review is positive" if prediction[1][0] > 0.5 else "Review is negative"

        return sentiment

