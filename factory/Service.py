import config
from service.MovieReviewPrediction import MovieReviewPrediction


class Service:

    def __init__(self):
        self.service_type = config.SERVICE_TYPE

    def get_service(self):
        if self.service_type == "ML":
            return MovieReviewPrediction()

