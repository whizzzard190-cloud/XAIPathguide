from flask import Flask, render_template, request
import sys
import os
sys.path.append(os.path.abspath("src"))

from recommend import recommend_courses

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = None

    if request.method == "POST":

        try:
            skill = int(request.form["skill"])
            goal = request.form["goal"]
            pref = request.form["preference"]

            exp_raw = request.form["experience"]
            exp = int(exp_raw) if exp_raw else 0

            learner = {
                "skill_level": skill,
                "goal": goal,
                "preference": pref,
                "experience_years": exp
            }

            recommendations = recommend_courses(learner)

            topic_map = {0: "python", 1: "ml", 2: "data", 3: "web"}
            recommendations["topic"] = recommendations["topic"].map(topic_map)


        except Exception as e:
            return f"Input error: {str(e)}"

    return render_template("index.html", recommendations=recommendations)
if __name__ == "__main__":
    app.run(debug=True)
