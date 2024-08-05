import matplotlib.pyplot as plt
import seaborn as sns
import logging

def plot_distribution(df, column, filename):
    try:
        sns.histplot(df[column], kde=True) 
        # replaced "distplot" with "histplot" as it has been deprecated
        plt.savefig(f'src/visualization/images/{filename}.png')
        plt.show()
        logging.info(f'Distribution plot for {column} displayed successfully.')
    except Exception as e:
        logging.error(f'Error displaying distribution plot for {column}: {e}')
        raise

def plot_loan_status(df, filename):
    try:
        df['Loan_Status'].value_counts().plot.bar()
        plt.savefig(f'src/visualization/images/{filename}.png')
        plt.show()
        logging.info('Loan status bar plot displayed successfully.')
    except Exception as e:
        logging.error(f'Error displaying loan status bar plot: {e}')
        raise
