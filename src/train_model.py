import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_model():

    os.makedirs("models", exist_ok=True)

    data = pd.read_csv("data/processed_data.csv")

    X = data.drop(columns=["rating"])
    y = data["rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Model MSE:", mean_squared_error(y_test, preds))

    joblib.dump(model, "models/recommender.pkl")

    print("Model saved.")

if __name__ == "__main__":
    train_model()