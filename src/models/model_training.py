# load best params from gridsearch and train the GradientBoostingRegressor model
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle

def load_training_data(X_train_path, y_train_path):
    X_train_scaled = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    return X_train_scaled, y_train

def load_best_params(params_path):
    with open(params_path, 'rb') as f:
        best_params = pickle.load(f)
    return best_params

if __name__ == "__main__":
    X_train_path = 'data/processed_data/X_train_scaled.csv'
    y_train_path = 'data/processed_data/y_train.csv'
    best_params_path = 'models/best_params.pkl'

    X_train_scaled, y_train = load_training_data(X_train_path, y_train_path)
    best_params = load_best_params(best_params_path)

    # Train the model with the best hyperparameters from GridSearch
    model = GradientBoostingRegressor(**best_params, random_state=42)
    model.fit(X_train_scaled, y_train.values.ravel())

    # Save the trained model
    with open('models/trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model trained with params:", best_params)
    print("Trained model saved to models/trained_model.pkl")
