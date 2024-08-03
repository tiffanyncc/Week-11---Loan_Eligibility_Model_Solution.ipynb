import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_data(df):
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    
    df = df.drop('Loan_ID', axis=1)
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], dtype=int)
    
    return df

def split_data(df, target_column='Loan_Status', test_size=0.2, random_state=123, stratify=True):
    if stratify:
        x_train, x_test, y_train, y_test = train_test_split(df.drop(target_column, axis=1), df[target_column], test_size=test_size, random_state=random_state, stratify=df[target_column])
    else:
        x_train, x_test, y_train, y_test = train_test_split(df.drop(target_column, axis=1), df[target_column], test_size=test_size, random_state=random_state)
    
    return x_train, x_test, y_train, y_test

def scale_data(x_train, x_test, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    return x_train_scaled, x_test_scaled
