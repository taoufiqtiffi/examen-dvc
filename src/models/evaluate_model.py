# load trained model and evaluate it on the test set, save predictions and metrics
import json
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score

def load_test_data(X_test_path, y_test_path):
    X_test_scaled = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    return X_test_scaled, y_test

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    X_test_path = 'data/processed_data/X_test_scaled.csv'
    y_test_path = 'data/processed_data/y_test.csv'
    model_path = 'models/trained_model.pkl'

    X_test_scaled, y_test = load_test_data(X_test_path, y_test_path)
    model = load_model(model_path)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Save predictions
    predictions_df = pd.DataFrame({'y_test': y_test.values.ravel(), 'y_pred': y_pred})
    predictions_df.to_csv('data/processed_data/predictions.csv', index=False)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    scores = {'mse': mse, 'rmse': rmse, 'r2': r2}

    with open('metrics/scores.json', 'w') as f:
        json.dump(scores, f, indent=4)

    print("Evaluation metrics:", scores)
    print("Predictions saved to data/processed_data/predictions.csv")
    print("Scores saved to metrics/scores.json")