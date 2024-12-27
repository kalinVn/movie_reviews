from preprocess.MovieReviewNLTK import MovieReviewNLTK
from preprocess.MovieReviewKeras import MovieReviewKeras
import config


class PreprocessFactory:

    def __init__(self, name):
        self.name = name

    def get_obj(self):
        if self.name == config.MOVIE_REVIEW_PREDICTION_NLTK:
            return MovieReviewNLTK()

        return MovieReviewKeras()

