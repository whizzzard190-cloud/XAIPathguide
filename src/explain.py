import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def explain_prediction():

    model = joblib.load("models/recommender.pkl")

    data = pd.read_csv("data/processed_data.csv")

    X = data.drop(columns=["rating"])

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X[:50])

    shap.summary_plot(shap_values, X[:50], show=False)

    plt.tight_layout()
    plt.savefig("models/shap_summary.png")

    print("SHAP explanation saved as models/shap_summary.png")

if __name__ == "__main__":
    explain_prediction()
