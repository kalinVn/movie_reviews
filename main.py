from factory.Service import Service


def movie_reviews_prediction():
    factory_service = Service()
    service = factory_service.get_service()

    service.preprocess()
    service.build()
    service.test_accuracy_score()


movie_reviews_prediction()






