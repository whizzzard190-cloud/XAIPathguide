import pandas as pd
import numpy as np
import random

np.random.seed(42)

# ---------- Learners ----------
learners = []

skills = ["python", "ml", "data", "web"]
goals = ["data_science", "ai_engineer", "web_dev"]

for i in range(50):
    learners.append({
        "learner_id": i,
        "skill_level": random.randint(1, 5),
        "goal": random.choice(goals),
        "preference": random.choice(skills),
        "experience_years": random.randint(0, 5)
    })

learners_df = pd.DataFrame(learners)

# ---------- Courses ----------
courses = []

topics = ["python", "ml", "data", "web"]

for i in range(20):
    courses.append({
        "course_id": i,
        "topic": random.choice(topics),
        "difficulty": random.randint(1, 5),
        "duration_hours": random.randint(5, 40)
    })

courses_df = pd.DataFrame(courses)

# ---------- Interactions ----------
interactions = []

for _ in range(200):
    interactions.append({
        "learner_id": random.randint(0, 49),
        "course_id": random.randint(0, 19),
        "rating": random.randint(1, 5)
    })

interactions_df = pd.DataFrame(interactions)

# Save datasets
learners_df.to_csv("data/learners.csv", index=False)
courses_df.to_csv("data/courses.csv", index=False)
interactions_df.to_csv("data/interactions.csv", index=False)

print("Datasets generated successfully!")
