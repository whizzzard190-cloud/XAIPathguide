from explain import explain_user
import pandas as pd
import joblib
import os
from train_model import train_model
course_names = {
    0:"Advanced Data Analytics",
    1:"Machine Learning Foundations",
    2:"Full Stack Web Development",
    3:"ML for Beginners",
    4:"Python Basics",
    5:"Intro to Data Analysis",
    6:"Advanced Python",
    7:"Frontend Essentials",
    8:"Python for AI",
    9:"Web Development Basics",
    10:"Python Data Engineering",
    11:"Applied Machine Learning",
    12:"Intermediate Machine Learning",
    13:"Python Automation",
    14:"ML Fundamentals",
    15:"Advanced ML Algorithms",
    16:"Data Processing",
    17:"Advanced Web Apps",
    18:"Deep Learning",
    19:"Python Projects"
}
reasons = {
        0: "Builds advanced analytics foundation for real-world data science careers.",
        1: "Introduces core machine learning concepts for intelligent system development.",
        2: "Teaches complete web stack for deploying real applications.",
        3: "Perfect beginner introduction to practical machine learning techniques.",
        4: "Strengthens Python fundamentals required across all AI workflows.",
        5: "Covers essential data analysis skills using Python tools.",
        6: "Advanced Python concepts for building scalable backend systems.",
        7: "Frontend essentials for responsive professional web interfaces.",
        8: "Python applied to AI models and automation pipelines.",
        9: "Beginner-friendly overview of modern web development practices.",
        10: "Python techniques used in data engineering pipelines.",
        11: "Hands-on applied machine learning for real datasets.",
        12: "Intermediate ML techniques for feature engineering and modeling.",
        13: "Automates repetitive tasks using Python scripting.",
        14: "Practical ML foundations using Google’s crash course.",
        15: "Advanced learning algorithms for professional ML engineers.",
        16: "Intermediate data processing for structured datasets.",
        17: "Advanced full-stack web application architecture.",
        18: "Deep learning applications for modern AI systems.",
        19: "Applied Python projects for portfolio development."
    }
sources = {
        0: [
            {"name": "Coursera", "link": "https://www.coursera.org/professional-certificates/google-advanced-data-analytics", "type": "Free to Learn", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/topic/data-analysis/expert/?utm_campaign=Search_Keyword_Beta_Prof_la.DE_cc.ROW-German&utm_source=google&utm_medium=paid-search&portfolio=ROW-German&utm_audience=mx&utm_tactic=nb&utm_term=data%20analysis%20courses%20for%20beginners&utm_content=g&funnel=&test=&gad_source=1&gad_campaignid=21485730605&gbraid=0AAAAADROdO2CyBRz6I68UA2nAicGl4xIH&gclid=Cj0KCQiAqeDMBhDcARIsAJEbU9TFdaSNOGpEhhvvv_EJfvqsGcU2U-gc28JEpCPbhp5l8rngh912WqYaAndCEALw_wcB", "type": "Paid", "best": False}
        ],
        5: [
            {"name": "Coursera", "link": "https://www.coursera.org/learn/data-analytics-introduction", "type": "Free to Learn", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/topic/data-analysis/free/?utm_campaign=Search_Keyword_Alpha_Prof_la.ES_cc.ROW-Spanish&utm_source=google&utm_medium=paid-search&portfolio=ROW-Spanish&utm_audience=mx&utm_tactic=nb&utm_term=capacitaci%C3%B3n%20data%20science&utm_content=g&funnel=&test=&gad_source=1&gad_campaignid=21487757262&gbraid=0AAAAADROdO35C6qdcLR9yERBOQxiEJaSn&gclid=Cj0KCQiAqeDMBhDcARIsAJEbU9QeJj16ifabent1TW-793pEc6Rfh-IpKDNnh3qCdilGPWLoFilpoYoaAjPaEALw_wcB", "type": "Paid", "best": False}
        ],
        16: [
            {"name": "freeCodeCamp", "link": "https://www.freecodecamp.org/learn/data-analysis-with-python", "type": "Free", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/course/data-engineering-essentials", "type": "Paid", "best": False}
        ],
        1: [
            {"name": "Coursera", "link": "https://www.coursera.org/learn/machine-learning", "type": "Free to Learn", "best": True},
            {"name": "DeepLearning.AI", "link": "https://www.deeplearning.ai/courses/machine-learning-specialization", "type": "Paid", "best": False}
        ],
        3: [
            {"name": "GitHub", "link": "https://github.com/microsoft/ML-For-Beginners", "type": "Free", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/course/machinelearning", "type": "Paid", "best": False}
        ],
        11: [
            {"name": "Kaggle", "link": "https://www.kaggle.com/learn/intro-to-machine-learning", "type": "Free", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/topic/machine-learning/free/?utm_campaign=Search_Keyword_Beta_Prof_la.ES_cc.ROW-Spanish&utm_source=google&utm_medium=paid-search&portfolio=ROW-Spanish&utm_audience=mx&utm_tactic=nb&utm_term=estudiar%20machine%20learning&utm_content=g&funnel=&test=&gad_source=1&gad_campaignid=21487757259&gbraid=0AAAAADROdO1QiNHjy75wMDopwTypr02Di&gclid=Cj0KCQiAqeDMBhDcARIsAJEbU9RvDr4CvSbCkiFN2fxujVmMAlzIdk31rQvaceiA5e4K8Z--KcjbndIaAjieEALw_wcB", "type": "Paid", "best": False}
        ],
        12: [
            {"name": "Kaggle", "link": "https://www.kaggle.com/learn/intermediate-machine-learning", "type": "Free", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/topic/machine-learning/free/?utm_campaign=Search_Keyword_Beta_Prof_la.ES_cc.ROW-Spanish&utm_source=google&utm_medium=paid-search&portfolio=ROW-Spanish&utm_audience=mx&utm_tactic=nb&utm_term=estudiar%20machine%20learning&utm_content=g&funnel=&test=&gad_source=1&gad_campaignid=21487757259&gbraid=0AAAAADROdO1QiNHjy75wMDopwTypr02Di&gclid=Cj0KCQiAqeDMBhDcARIsAJEbU9TJ-ZCphcQof6fkyNWT11BFvYirF_bTUl99nZRVs5ba8IlQ-oUbmfsaAjr_EALw_wcB", "type": "Paid", "best": False}
        ],
        14: [
            {"name": "Google", "link": "https://developers.google.com/machine-learning/crash-course", "type": "Free", "best": True},
            {"name": "Coursera", "link": "https://www.coursera.org/courses?query=free&skills=Machine%20Learning", "type": "Paid", "best": False}
        ],
        15: [
            {"name": "Coursera", "link": "https://www.coursera.org/learn/advanced-learning-algorithms", "type": "Free to Learn", "best": True},
            {"name": "Coursera", "link": "https://www.coursera.org/specializations/deep-learning", "type": "Paid", "best": False}
        ],
        18: [
            {"name": "fast.ai", "link": "https://course.fast.ai", "type": "Free", "best": True},
            {"name": "DeepLearning.AI", "link": "https://www.deeplearning.ai/courses/deep-learning-specialization", "type": "Paid", "best": False}
        ],
        4: [
            {"name": "freeCodeCamp", "link": "https://www.freecodecamp.org/learn/scientific-computing-with-python", "type": "Free", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/course/complete-python-bootcamp", "type": "Paid", "best": False}
        ],
        6: [
            {"name": "RealPython", "link": "https://realpython.com", "type": "Paid", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/course/advanced-python-programming", "type": "Paid", "best": False}
        ],
        8: [
            {"name": "freeCodeCamp", "link": "https://www.freecodecamp.org/learn/machine-learning-with-python", "type": "Free", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp", "type": "Paid", "best": False}
        ],
        10: [
            {"name": "Kaggle", "link": "https://www.kaggle.com/learn/python", "type": "Free", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/course/data-engineering-with-python", "type": "Paid", "best": False}
        ],
        13: [
            {"name": "Automate", "link": "https://automatetheboringstuff.com", "type": "Free", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/course/automate", "type": "Paid", "best": False}
        ],
        19: [
            {"name": "freeCodeCamp", "link": "https://www.freecodecamp.org/learn/scientific-computing-with-python", "type": "Free", "best": True},
            {"name": "Coursera", "link": "https://www.coursera.org/projects/python-project", "type": "Paid", "best": False}
        ],
        2: [
            {"name": "freeCodeCamp", "link": "https://www.freecodecamp.org/learn", "type": "Free", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/course/the-complete-web-development-bootcamp", "type": "Paid", "best": False}
        ],
        7: [
            {"name": "freeCodeCamp", "link": "https://www.freecodecamp.org/learn/responsive-web-design", "type": "Free", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/course/the-complete-javascript-course", "type": "Paid", "best": False}
        ],
        9: [
            {"name": "MDN", "link": "https://developer.mozilla.org/en-US/docs/Learn", "type": "Free", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/course/web-development-for-beginners", "type": "Paid", "best": False}
        ],
        17: [
            {"name": "FullStackOpen", "link": "https://fullstackopen.com/en/", "type": "Free", "best": True},
            {"name": "Udemy", "link": "https://www.udemy.com/topic/web-development/free/?utm_campaign=Search_Keyword_Gamma_NonP_la.ES_cc.ROW-Spanish&utm_source=google&utm_medium=paid-search&portfolio=ROW-Spanish&utm_audience=mx&utm_tactic=nb&utm_term=mejores%20cursos%20de%20programaci%C3%B3n&utm_content=g&funnel=&test=&gad_source=1&gad_campaignid=21487757256&gbraid=0AAAAADROdO3eVaYstPTTz3qJ5LBehu4H9&gclid=Cj0KCQiAqeDMBhDcARIsAJEbU9TF4STP6uKHnMCj5WrpVAwFN9eGqslylipj4WgII5A-EzezecjnM2saAp75EALw_wcB", "type": "Paid", "best": False}
        ]
    }
def recommend_courses(learner_profile, top_n=5):

    learner_profile["experience_years"] = max(0, learner_profile["experience_years"])

    if not os.path.exists("models/recommender.pkl"):
        train_model()

    model = joblib.load("models/recommender.pkl")

    le_goal = joblib.load("models/le_goal.pkl")
    le_preference = joblib.load("models/le_preference.pkl")
    le_topic = joblib.load("models/le_topic.pkl")

    courses = pd.read_csv("data/courses.csv")

    courses["topic_display"] = courses["topic"]
    courses["topic_encoded"] = le_topic.transform(courses["topic"])

    df = courses.copy()
    df["learner_id"] = 0

    df["skill_level"] = learner_profile["skill_level"]
    df["goal"] = le_goal.transform([learner_profile["goal"]])[0]
    df["preference"] = le_preference.transform([learner_profile["preference"]])[0]
    df["experience_years"] = learner_profile["experience_years"]

    df = df[[
        "learner_id","course_id","skill_level","goal",
        "preference","experience_years","topic_encoded",
        "difficulty","duration_hours"
    ]]

    df.rename(columns={"topic_encoded":"topic"}, inplace=True)

    preds = model.predict(df)
    courses["score"] = preds

    # ---------------- GOAL BOOST ----------------

    goal = learner_profile["goal"]

    if goal == "data_scientist":
        courses.loc[courses.topic_display=="data","score"] += 1.5
        courses.loc[courses.topic_display=="ml","score"] += 1.0

    elif goal in ["ml_engineer","ai_engineer"]:
        courses.loc[courses.topic_display=="ml","score"] += 1.5

    elif goal in ["web_developer","frontend_developer","backend_developer","fullstack_developer"]:
        courses.loc[courses.topic_display=="web","score"] += 2.0

    elif goal == "devops_engineer":
        courses.loc[courses.topic_display=="python","score"] += 1.2
        courses.loc[courses.topic_display=="data","score"] += 0.8

    elif goal in ["data_analyst","business_analyst"]:
        courses.loc[courses.topic_display=="data","score"] += 1.5

    # ---------------- FINAL RANK ----------------

    ranked = courses.sort_values("score", ascending=False)
    top_courses = ranked.head(top_n).copy()
    # Curriculum ordering (easy → hard)
    top_courses = top_courses.sort_values("difficulty")
    top_courses["topic"] = top_courses["topic_display"]
    top_courses["course_name"] = top_courses["course_id"].map(course_names)

    top_courses.attrs["learning_path"] = " → ".join(top_courses["course_name"])

    top_courses["reason"] = top_courses["course_id"].apply(lambda x: reasons.get(x))
    top_courses["sources"] = top_courses["course_id"].apply(lambda x: sources.get(x,[]))

    # ---------------- SHAP ----------------

    try:
        xai = explain_user(df.iloc[[preds.argmax()]])
        top_courses.attrs["xai"] = xai
    except:
        top_courses.attrs["xai"] = {}

    return top_courses