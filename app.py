import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Student Performance ML System",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# REAL DATASET
# ============================================================
DATASET_URL = (
    "https://raw.githubusercontent.com/desithomas/data_607_assignment_1/"
    "refs/heads/main/StudentPerformanceFactors.csv"
)
LOCAL_DATASET = "StudentPerformanceFactors.csv"


@st.cache_data
def load_data():
    try:
        # Loads the real 6,607-row Student Performance Factors dataset.
        data = pd.read_csv(DATASET_URL)
        return data, "Online source"
    except Exception:
        # Allows the app to work offline if the CSV has been downloaded
        # and placed beside app.py.
        if os.path.exists(LOCAL_DATASET):
            return pd.read_csv(LOCAL_DATASET), "Local CSV"
        raise RuntimeError(
            "Dataset could not be loaded. Connect to the internet or place "
            "'StudentPerformanceFactors.csv' in the same folder as app.py."
        )


df, data_source = load_data()

# ============================================================
# DATA CLEANING
# ============================================================
# The real dataset has 6,607 rows and 20 columns.
# Three categorical columns contain some missing values.
categorical_columns = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"
]

numeric_columns = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity"
]

target = "Exam_Score"

# ============================================================
# SUPERVISED LEARNING
# ============================================================
X = df[categorical_columns + numeric_columns]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ============================================================
# UNSUPERVISED LEARNING
# ============================================================
# K-Means uses the six numeric academic/lifestyle features.
cluster_data = df[numeric_columns].copy()
cluster_data = cluster_data.fillna(cluster_data.median())

scaler = StandardScaler()
scaled_cluster_data = scaler.fit_transform(cluster_data)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_cluster_data)

cluster_df = df.copy()
cluster_df["Cluster"] = clusters

# Give clusters understandable names based on their average exam score.
cluster_means = (
    cluster_df.groupby("Cluster")[target]
    .mean()
    .sort_values()
)

cluster_names = [
    "Needs Improvement",
    "Average",
    "High Performing"
]

cluster_labels = {
    cluster_id: cluster_names[i]
    for i, cluster_id in enumerate(cluster_means.index)
}

cluster_df["Performance_Group"] = cluster_df["Cluster"].map(cluster_labels)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🎓 ML Module")
st.sidebar.caption("Supervised + Unsupervised Learning")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📈 Supervised Learning",
        "🔵 Unsupervised Learning",
        "📊 Dataset"
    ]
)

# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":

    st.title("🎓 Student Performance ML System")
    st.subheader("CCE: ML Module based on Supervised and Unsupervised Learning")

    st.write(
        "This project uses the real **Student Performance Factors** dataset "
        "containing 6,607 student records and 20 columns. It demonstrates "
        "both supervised and unsupervised machine learning."
    )

    st.info(
        f"Dataset loaded: **{len(df):,} students × {df.shape[1]} columns** "
        f"({data_source})"
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Students", f"{len(df):,}")

    with c2:
        st.metric("Columns", df.shape[1])

    with c3:
        st.metric("Algorithms", "2")

    with c4:
        st.metric("Target", "Exam Score")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Supervised Learning")
        st.write(
            "Random Forest Regression learns from previous student records "
            "where the exam score is already known, then predicts the exam "
            "score for a new student."
        )
        st.success("Output: Predicted Exam Score")

    with col2:
        st.markdown("### 🔵 Unsupervised Learning")
        st.write(
            "K-Means Clustering finds groups of students with similar "
            "academic and study-related characteristics without using "
            "the exam score to create the groups."
        )
        st.success("Output: Student Performance Groups")

    st.divider()

    st.markdown("### 🔄 Project Workflow")

    st.code("""
Real Student Dataset (6,607 records)
                ↓
         Data Preprocessing
                ↓
       ┌────────┴─────────┐
       ↓                  ↓
 Supervised           Unsupervised
 Learning             Learning
       ↓                  ↓
 Random Forest         K-Means
 Regression           Clustering
       ↓                  ↓
 Predict Score       Find Groups
       └────────┬─────────┘
                ↓
        Streamlit Dashboard
""")

    st.caption(
        "Source: Kaggle's Student Performance Factors dataset; the app "
        "loads a public copy of the same 6,607 × 20 CSV."
    )

# ============================================================
# SUPERVISED
# ============================================================
elif page == "📈 Supervised Learning":

    st.title("📈 Supervised Learning")
    st.write("Algorithm: **Random Forest Regression**")

    st.markdown("### Model Performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("R² Score", f"{r2:.3f}")

    with c2:
        st.metric("MAE", f"{mae:.2f}")

    with c3:
        st.metric("RMSE", f"{rmse:.2f}")

    st.divider()

    st.markdown("### 🎯 Predict a Student's Exam Score")

    st.write(
        "Enter the student's academic and study-related information. "
        "The Random Forest model predicts the expected exam score."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        hours_studied = st.slider(
            "Hours Studied per Week", 1, 44, 20
        )
        attendance = st.slider(
            "Attendance (%)", 60, 100, 80
        )

    with col2:
        sleep_hours = st.slider(
            "Sleep Hours per Day", 4, 10, 7
        )
        previous_scores = st.slider(
            "Previous Scores", 50, 100, 75
        )

    with col3:
        tutoring = st.slider(
            "Tutoring Sessions", 0, 8, 1
        )
        physical_activity = st.slider(
            "Physical Activity (hours/week)", 0, 6, 3
        )

    st.markdown("#### Other Student Factors")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        parental = st.selectbox(
            "Parental Involvement",
            ["Low", "Medium", "High"],
            index=1
        )
        resources = st.selectbox(
            "Access to Resources",
            ["Low", "Medium", "High"],
            index=1
        )

    with c2:
        extracurricular = st.selectbox(
            "Extracurricular Activities",
            ["No", "Yes"],
            index=1
        )
        motivation = st.selectbox(
            "Motivation Level",
            ["Low", "Medium", "High"],
            index=1
        )

    with c3:
        internet = st.selectbox(
            "Internet Access",
            ["No", "Yes"],
            index=1
        )
        family_income = st.selectbox(
            "Family Income",
            ["Low", "Medium", "High"],
            index=1
        )

    with c4:
        teacher_quality = st.selectbox(
            "Teacher Quality",
            ["Low", "Medium", "High"],
            index=1
        )
        school_type = st.selectbox(
            "School Type",
            ["Public", "Private"],
            index=0
        )

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        peer_influence = st.selectbox(
            "Peer Influence",
            ["Negative", "Neutral", "Positive"],
            index=1
        )

    with c6:
        learning_disability = st.selectbox(
            "Learning Disabilities",
            ["No", "Yes"],
            index=0
        )

    with c7:
        parental_education = st.selectbox(
            "Parental Education Level",
            ["High School", "College", "Postgraduate"],
            index=0
        )

    with c8:
        distance_home = st.selectbox(
            "Distance from Home",
            ["Near", "Moderate", "Far"],
            index=0
        )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    input_data = pd.DataFrame({
        "Hours_Studied": [hours_studied],
        "Attendance": [attendance],
        "Sleep_Hours": [sleep_hours],
        "Previous_Scores": [previous_scores],
        "Tutoring_Sessions": [tutoring],
        "Physical_Activity": [physical_activity],
        "Parental_Involvement": [parental],
        "Access_to_Resources": [resources],
        "Extracurricular_Activities": [extracurricular],
        "Motivation_Level": [motivation],
        "Internet_Access": [internet],
        "Family_Income": [family_income],
        "Teacher_Quality": [teacher_quality],
        "School_Type": [school_type],
        "Peer_Influence": [peer_influence],
        "Learning_Disabilities": [learning_disability],
        "Parental_Education_Level": [parental_education],
        "Distance_from_Home": [distance_home],
        "Gender": [gender]
    })

    if st.button("🔮 Predict Exam Score", type="primary"):
        prediction = rf_model.predict(input_data)[0]
        prediction = np.clip(prediction, 0, 100)

        st.success(
            f"Predicted Exam Score: **{prediction:.1f} / 100**"
        )

        if prediction >= 75:
            st.info("Performance Category: High")
        elif prediction >= 50:
            st.warning("Performance Category: Average")
        else:
            st.error("Performance Category: Needs Improvement")

    st.divider()

    st.markdown("### 📊 Actual vs Predicted Exam Scores")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, alpha=0.55)

    minimum = min(y_test.min(), y_pred.min())
    maximum = max(y_test.max(), y_pred.max())

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    ax.set_xlabel("Actual Exam Score")
    ax.set_ylabel("Predicted Exam Score")
    ax.set_title("Actual vs Predicted Exam Scores")
    ax.grid(alpha=0.25)

    st.pyplot(fig)

# ============================================================
# UNSUPERVISED
# ============================================================
elif page == "🔵 Unsupervised Learning":

    st.title("🔵 Unsupervised Learning")
    st.write("Algorithm: **K-Means Clustering**")

    st.write(
        "K-Means groups students based on similarities in their study "
        "and academic characteristics. Exam Score is NOT used to create "
        "the clusters."
    )

    st.divider()

    st.markdown("### 👥 Student Groups")

    group_summary = (
        cluster_df.groupby("Performance_Group")
        .agg(
            Students=("Performance_Group", "count"),
            Avg_Study_Hours=("Hours_Studied", "mean"),
            Avg_Attendance=("Attendance", "mean"),
            Avg_Previous_Scores=("Previous_Scores", "mean"),
            Avg_Exam_Score=("Exam_Score", "mean")
        )
        .reindex([
            "Needs Improvement",
            "Average",
            "High Performing"
        ])
    )

    st.dataframe(
        group_summary.round(2),
        use_container_width=True
    )

    st.divider()

    st.markdown("### 📍 Cluster Visualization")

    x_axis = st.selectbox(
        "X-axis",
        numeric_columns,
        index=0
    )

    y_axis = st.selectbox(
        "Y-axis",
        numeric_columns,
        index=1
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for group in [
        "Needs Improvement",
        "Average",
        "High Performing"
    ]:
        subset = cluster_df[
            cluster_df["Performance_Group"] == group
        ]

        ax.scatter(
            subset[x_axis],
            subset[y_axis],
            label=group,
            alpha=0.55
        )

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title("K-Means Student Clusters")
    ax.legend()
    ax.grid(alpha=0.25)

    st.pyplot(fig)

    st.divider()

    st.markdown("### 🔎 Clustered Student Data")

    display_columns = [
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Scores",
        "Tutoring_Sessions",
        "Physical_Activity",
        "Exam_Score",
        "Performance_Group"
    ]

    st.dataframe(
        cluster_df[display_columns],
        use_container_width=True
    )

# ============================================================
# DATASET
# ============================================================
elif page == "📊 Dataset":

    st.title("📊 Real Student Performance Factors Dataset")

    st.write(
        f"The application is using **{len(df):,} student records** "
        f"and **{df.shape[1]} columns**."
    )

    st.dataframe(df, use_container_width=True)

    st.divider()

    st.markdown("### Dataset Statistics")
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

    st.markdown("### 📈 Exam Score Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["Exam_Score"], bins=15)
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Exam Scores")
    st.pyplot(fig)

st.sidebar.divider()
st.sidebar.caption("CCE ML Project • Real Student Performance Factors Dataset")
