import shap
import joblib
import matplotlib

# Force headless backend (NO TKINTER)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def explain_user(df):

    model = joblib.load("models/recommender.pkl")

    # Load background for SHAP (small sample)
    background = pd.read_csv("data/processed_data.csv").drop(columns=["rating"]).sample(50)

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional"
    )

    shap_exp = explainer(df, check_additivity=False)

    # Waterfall plot
    shap.plots.waterfall(shap_exp[0], show=False)

    plt.tight_layout()
    plt.savefig("models/shap_user.png", dpi=120)
    plt.close()

    # Text explanation
    values = shap_exp[0].values
    features = df.columns

    explanation = dict(zip(features, values))

    explanation = dict(
        sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    # Remove learner_id from display
    explanation.pop("learner_id", None)

    return explanation