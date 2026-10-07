# Student Performance ML System — Real Dataset Version

CCE project demonstrating supervised and unsupervised machine learning using the real **Student Performance Factors** dataset.

## Dataset

- Rows: **6,607**
- Columns: **20**
- Target: **Exam_Score**
- Original source: Kaggle — Student Performance Factors
- Kaggle dataset: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
- Public CSV copy used by the app:
  https://raw.githubusercontent.com/desithomas/data_607_assignment_1/refs/heads/main/StudentPerformanceFactors.csv

The dataset contains 6,607 student records and 20 columns, including study hours, attendance, previous scores, sleep hours, tutoring sessions, lifestyle/family/school factors, and exam score.

## Machine Learning

### Supervised Learning
Algorithm: **Random Forest Regression**

The model predicts `Exam_Score`.

The application:
- handles missing values
- one-hot encodes categorical variables
- trains on 80% of the data
- tests on 20% of the data
- reports R², MAE and RMSE
- provides an interactive prediction form

### Unsupervised Learning
Algorithm: **K-Means Clustering**

The clustering model uses:
- Hours_Studied
- Attendance
- Sleep_Hours
- Previous_Scores
- Tutoring_Sessions
- Physical_Activity

The target `Exam_Score` is NOT used to create clusters.

The app creates 3 clusters and gives them understandable labels based on their average exam scores:
- Needs Improvement
- Average
- High Performing

## How to Run

### 1. Install Python

Python 3.10+ is recommended.

### 2. Open the project folder

```bash
cd ML_CCE_Project
```

### 3. Install libraries

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit

```bash
streamlit run app.py
```

The application normally opens at:

```text
http://localhost:8501
```

## Offline option

The app normally downloads the dataset from the public CSV URL automatically.

If internet access is unavailable, download the dataset as:

```text
StudentPerformanceFactors.csv
```

and place it in the same folder as `app.py`. The app will then use the local CSV.

## Project Structure

```text
ML_CCE_Project/
│
├── app.py
├── requirements.txt
├── README.md
└── StudentPerformanceFactors.csv   # optional local copy
```
