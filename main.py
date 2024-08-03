import logging
import matplotlib.pyplot as plt
from src.data.make_dataset import load_data, save_data
from src.features.build_features import preprocess_data, split_data, scale_data
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.visualization.visualize import plot_distribution, plot_loan_status
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    try:
        df = load_data('src/data/raw/credit.csv')
        logging.info('Data loaded successfully.')
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        return

    try:
        plot_loan_status(df)
        logging.info('Loan status distribution plotted.')
    except Exception as e:
        logging.error(f'Error plotting loan status distribution: {e}')
    
    try:
        plot_distribution(df, 'LoanAmount')
        logging.info('Loan amount distribution plotted.')
    except Exception as e:
        logging.error(f'Error plotting loan amount distribution: {e}')
    
    try:
        df = preprocess_data(df)
        logging.info('Data preprocessed successfully.')
    except Exception as e:
        logging.error(f'Error preprocessing data: {e}')
        return

    try:
        save_data(df, 'src/data/processed/Processed_Credit_Dataset.csv')
        logging.info('Processed data saved successfully.')
    except Exception as e:
        logging.error(f'Error saving processed data: {e}')
        return

    try:
        x_train, x_test, y_train, y_test = split_data(df)
        logging.info('Data split into training and testing sets.')
    except Exception as e:
        logging.error(f'Error splitting data: {e}')
        return

    try:
        x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
        logging.info('Data scaled successfully.')
    except Exception as e:
        logging.error(f'Error scaling data: {e}')
        return
    
    # Train and evaluate Logistic Regression Model
    try:
        lr_model = LogisticRegressionModel()
        lr_model.train(x_train_scaled, y_train)
        accuracy, conf_matrix = lr_model.evaluate(x_test_scaled, y_test)
        logging.info(f'Logistic Regression - Accuracy: {accuracy}, Confusion Matrix: {conf_matrix}')

        # Cross-validation for Logistic Regression
        kfold = KFold(n_splits=5)
        lr_scores = cross_val_score(lr_model.model, x_train_scaled, y_train, cv=kfold)
        logging.info(f'Logistic Regression - Cross-validation accuracy scores: {lr_scores}')
        logging.info(f'Logistic Regression - Mean accuracy: {lr_scores.mean()}')
        logging.info(f'Logistic Regression - Standard deviation: {lr_scores.std()}')
    except Exception as e:
        logging.error(f'Error with Logistic Regression model: {e}')
    
    # Train and evaluate Random Forest Model
    try:
        rf_model = RandomForestModel()
        rf_model.train(x_train_scaled, y_train)
        accuracy, conf_matrix = rf_model.evaluate(x_test_scaled, y_test)
        logging.info(f'Random Forest - Accuracy: {accuracy}, Confusion Matrix: {conf_matrix}')

        # Cross-validation for Random Forest
        rf_scores = cross_val_score(rf_model.model, x_train_scaled, y_train, cv=kfold)
        logging.info(f'Random Forest - Cross-validation accuracy scores: {rf_scores}')
        logging.info(f'Random Forest - Mean accuracy: {rf_scores.mean()}')
        logging.info(f'Random Forest - Standard deviation: {rf_scores.std()}')
    except Exception as e:
        logging.error(f'Error with Random Forest model: {e}')

if __name__ == '__main__':
    main()