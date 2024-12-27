from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

class Visualizator:

    def plt_sgd_score(model):
        plt.plot(range(1, 11), model.history['acc'], label='Training Accuracy')
        plt.plot(range(1, 11), model.history['val_acc'], label='Validation Accuracy')
        plt.axis([1, 10, 0, 1])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Train and validate accuracy using sgd optimizer')
        plt.legend()
        plt.show()

    def show_confusion_matrix(model, x_test_padded, y_test):

        y_test_prediction = model.predict(x_test_padded)

        c_matrix = confusion_matrix(y_test, y_test_prediction)
        # ax = sns.heatmap(c_matrix, annot=True, xticklabels=["Negative Sentiment", "Positive sentiment"],
        #                  yticklabels=["Negative Sentiment", "Positive sentiment"], cbar=False, cmap="Blues", fmt='g')
        # ax.set_xlabel("Prediction")
        # ax.set_xyabel("Actual")
        # plt.show


