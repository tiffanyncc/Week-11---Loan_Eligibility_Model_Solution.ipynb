import matplotlib.pyplot as plt
import seaborn as sns
import logging

def plot_distribution(df, column):
    try:
        sns.histplot(df[column], kde=True) 
        # replaced "distplot" with "histplot" as it has been deprecated
        plt.show()
        logging.info(f'Distribution plot for {column} displayed successfully.')
    except Exception as e:
        logging.error(f'Error displaying distribution plot for {column}: {e}')
        raise

def plot_loan_status(df):
    try:
        df['Loan_Status'].value_counts().plot.bar()
        plt.show()
        logging.info('Loan status bar plot displayed successfully.')
    except Exception as e:
        logging.error(f'Error displaying loan status bar plot: {e}')
        raise
