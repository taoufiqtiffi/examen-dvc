# load raw data and split into train and test sets

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path, test_size=0.2, random_state=42, target_column='silica_concentrate'):
    # Load the raw data
    data = pd.read_csv(file_path)
    
    # Split the data into features and target variable
    X = data.drop([target_column, 'date'], axis=1)
    y = data[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = 'data/raw_data/raw.csv'
    X_train, X_test, y_train, y_test = load_and_split_data(file_path)

    # save 4 sets to csv files
    X_train.to_csv('data/processed_data/X_train.csv', index=False)
    X_test.to_csv('data/processed_data/X_test.csv', index=False)
    y_train.to_csv('data/processed_data/y_train.csv', index=False)
    y_test.to_csv('data/processed_data/y_test.csv', index=False)
    
    print("Training set size:", X_train.shape[0])
    print("Testing set size:", X_test.shape[0])