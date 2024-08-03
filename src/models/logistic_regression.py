from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

class LogisticRegressionModel:
    def __init__(self):
        try:
            self.model = LogisticRegression()
            logging.info('LogisticRegression initialized.')
        except Exception as e:
            logging.error(f'Error initializing LogisticRegression: {e}')
            raise

    def train(self, x_train, y_train):
        try:
            self.model.fit(x_train, y_train)
            logging.info('Logistic Regression model trained.')
        except Exception as e:
            logging.error(f'Error training Logistic Regression model: {e}')
            raise

    def evaluate(self, x_test, y_test):
        try:
            predictions = self.model.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            conf_matrix = confusion_matrix(y_test, predictions)
            logging.info('Logistic Regression model evaluated.')
            return accuracy, conf_matrix
        except Exception as e:
            logging.error(f'Error evaluating Logistic Regression model: {e}')
            raise

    def predict_proba(self, x_test):
        try:
            probas = self.model.predict_proba(x_test)
            logging.info('Logistic Regression probability predictions generated.')
            return probas
        except Exception as e:
            logging.error(f'Error generating probability predictions: {e}')
            raise
