from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
import config


class Classifier:

    def __init__(self):
        self.name = config.MODEL

    def get_model(self):
        if self.name == "svm":
            return svm.SVC()
        elif self.name == "logistic_regression":
            return LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
        elif self.name == "linear_regression":
            return LinearRegression()
        else:
            return Lasso()

