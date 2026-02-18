import pandas as pd
import joblib

def recommend_courses(learner_profile, top_n=5):

    model = joblib.load("models/recommender.pkl")

    le_goal = joblib.load("models/le_goal.pkl")
    le_preference = joblib.load("models/le_preference.pkl")
    le_topic = joblib.load("models/le_topic.pkl")

    courses = pd.read_csv("data/courses.csv")

    # Encode course topics
    courses["topic"] = le_topic.transform(courses["topic"])

    df = courses.copy()

    df["learner_id"] = 0

    df["skill_level"] = learner_profile["skill_level"]
    df["goal"] = le_goal.transform([learner_profile["goal"]])[0]
    df["preference"] = le_preference.transform([learner_profile["preference"]])[0]
    df["experience_years"] = learner_profile["experience_years"]

    df = df[[
        "learner_id",
        "course_id",
        "skill_level",
        "goal",
        "preference",
        "experience_years",
        "topic",
        "difficulty",
        "duration_hours"
    ]]

    preds = model.predict(df)

    courses["score"] = preds

    ranked = courses.sort_values(by="score", ascending=False)

    return ranked.head(top_n)

if __name__ == "__main__":

    sample_learner = {
        "skill_level": 3,
        "goal": "data_science",
        "preference": "python",
        "experience_years": 2
    }

    recs = recommend_courses(sample_learner)
    print(recs)
