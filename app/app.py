from flask import Flask, render_template, request
import sys
import os
import shutil

sys.path.append(os.path.abspath("src"))

from recommend import recommend_courses
from professions import PROFESSION_MAP

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = None
    xai = {}   # <-- IMPORTANT

    if request.method == "POST":

        try:
            # ----------------------------
            # Read form inputs
            # ----------------------------
            skill = int(request.form["skill"])
            profession = request.form["goal"]
            exp_raw = request.form["experience"]
            exp = max(0, int(exp_raw)) if exp_raw else 0

            # ----------------------------
            # Profession Mapping
            # ----------------------------
            mapped = PROFESSION_MAP.get(profession)

            if not mapped:
                return "Invalid profession selected."

            learner = {
                "skill_level": skill,
                "goal": mapped["goal"],
                "preference": mapped["preference"],
                "experience_years": exp
            }

            # ----------------------------
            # Get recommendations
            # ----------------------------
            recommendations = recommend_courses(learner)

            # ----------------------------
            # Get XAI from recommender
            # ----------------------------
            xai = recommendations.attrs.get("xai", {})

            # ----------------------------
            # Copy SHAP image
            # ----------------------------
            if os.path.exists("models/shap_user.png"):
                shutil.copy("models/shap_user.png", "app/static/shap_user.png")

        except Exception as e:
            return f"Input error: {str(e)}"

    return render_template(
        "index.html",
        recommendations=recommendations,
        xai=xai     # <-- PASSED TO HTML
    )


if __name__ == "__main__":
    app.run(debug=True)