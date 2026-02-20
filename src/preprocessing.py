import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data():

    os.makedirs("models", exist_ok=True)

    learners = pd.read_csv("data/learners.csv")
    courses = pd.read_csv("data/courses.csv")
    interactions = pd.read_csv("data/interactions.csv")

    data = interactions.merge(learners, on="learner_id")
    data = data.merge(courses, on="course_id")

    le_goal = LabelEncoder()
    le_preference = LabelEncoder()
    le_topic = LabelEncoder()

    data["goal"] = le_goal.fit_transform(data["goal"])
    data["preference"] = le_preference.fit_transform(data["preference"])
    data["topic"] = le_topic.fit_transform(data["topic"])

    joblib.dump(le_goal, "models/le_goal.pkl")
    joblib.dump(le_preference, "models/le_preference.pkl")
    joblib.dump(le_topic, "models/le_topic.pkl")

    data.to_csv("data/processed_data.csv", index=False)

    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_data()