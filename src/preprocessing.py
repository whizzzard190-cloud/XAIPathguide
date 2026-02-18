import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # Load datasets
    learners = pd.read_csv("data/learners.csv")
    courses = pd.read_csv("data/courses.csv")
    interactions = pd.read_csv("data/interactions.csv")

    # Merge datasets
    data = interactions.merge(learners, on="learner_id")
    data = data.merge(courses, on="course_id")

    # Encode categorical features
    le_goal = LabelEncoder()
    le_preference = LabelEncoder()
    le_topic = LabelEncoder()

    data["goal"] = le_goal.fit_transform(data["goal"])
    data["preference"] = le_preference.fit_transform(data["preference"])
    data["topic"] = le_topic.fit_transform(data["topic"])

    # Save encoders for later use (important for Flask)
    import joblib
    joblib.dump(le_goal, "models/le_goal.pkl")
    joblib.dump(le_preference, "models/le_preference.pkl")
    joblib.dump(le_topic, "models/le_topic.pkl")

    # Save processed data
    data.to_csv("data/processed_data.csv", index=False)

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_data()
