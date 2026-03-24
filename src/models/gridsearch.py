# load scaled data and choose a model, then apply GridSearchCV to find the best hyperparameters
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pickle

def load_scaled_data(X_train_path, y_train_path):
    # Load the scaled data
    X_train_scaled = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    return X_train_scaled, y_train

if __name__ == "__main__":
    X_train_scaled_path = 'data/processed_data/X_train_scaled.csv'
    y_train_path = 'data/processed_data/y_train.csv'
    
    X_train_scaled, y_train = load_scaled_data(X_train_scaled_path, y_train_path)
    
    # Define the model
    model = GradientBoostingRegressor(random_state=42)
    
    # Define the hyperparameters to search
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    
    # Fit GridSearchCV to the training data
    grid_search.fit(X_train_scaled, y_train.values.ravel())
    
    # save the best parameters to a pkl file
    with open('models/best_params.pkl', 'wb') as f:
        pickle.dump(grid_search.best_params_, f)
    
    # Print the best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)