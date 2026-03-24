# load splitted data and apply normalization (StandardScaler)

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_normalize_data(X_train_path, X_test_path, y_train_path, y_test_path):
    # Load the splitted data
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler on the training data and transform both training and testing data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    X_train_path = 'data/processed_data/X_train.csv'
    X_test_path = 'data/processed_data/X_test.csv'
    y_train_path = 'data/processed_data/y_train.csv'
    y_test_path = 'data/processed_data/y_test.csv'

    X_train_scaled, X_test_scaled, y_train, y_test = load_and_normalize_data(X_train_path, X_test_path, y_train_path, y_test_path)

    # Save the normalized data to new CSV files
    import pandas as pd
    pd.DataFrame(X_train_scaled).to_csv('data/processed_data/X_train_scaled.csv', index=False)
    pd.DataFrame(X_test_scaled).to_csv('data/processed_data/X_test_scaled.csv', index=False)
    
    print("Data normalization completed. Scaled training and testing sets saved.")