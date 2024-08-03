from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

class RandomForestModel:
    def __init__(self, n_estimators=100, min_samples_leaf=5, max_features='sqrt'):
        try:
            self.model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features)
            logging.info('RandomForestClassifier initialized.')
        except Exception as e:
            logging.error(f'Error initializing RandomForestClassifier: {e}')
            raise

    def train(self, x_train, y_train):
        try:
            self.model.fit(x_train, y_train)
            logging.info('Random Forest model trained.')
        except Exception as e:
            logging.error(f'Error training Random Forest model: {e}')
            raise

    def evaluate(self, x_test, y_test):
        try:
            predictions = self.model.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            conf_matrix = confusion_matrix(y_test, predictions)
            logging.info('Random Forest model evaluated.')
            return accuracy, conf_matrix
        except Exception as e:
            logging.error(f'Error evaluating Random Forest model: {e}')
            raise
