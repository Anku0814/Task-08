
# Task 08 – Improving Titanic Survival Predictions through Feature Engineering

This repository contains the solution for **Data Analytics Internship – Task 08: Improving Predictions through Feature Engineering** using the **Kaggle Titanic: Machine Learning from Disaster** dataset.

The goal is to apply effective **feature engineering** techniques and build a classification model that predicts whether a passenger survived (`Survived` = 0 or 1).

## Project Objectives

- Explore and understand the Titanic dataset.
- Handle **missing values** properly.
- Create new, meaningful features such as:
  - `Title` (extracted from passenger name)
  - `FamilySize`
  - `IsAlone`
  - `CabinKnown`
- Encode categorical variables using **One-Hot Encoding**.
- Train a **RandomForestClassifier** to predict survival.
- Evaluate the model using **Accuracy** and a **classification report**.
- Visualize **feature importance** for the engineered features.

## Repository Structure

```text
.
├── data/
│   └── train.csv                 # <- Kaggle Titanic training data (place here)
├── outputs/
│   └── feature_importances.png   # <- Generated automatically after running the script
├── titanic_feature_engineering.py
├── requirements.txt
└── README.md
```

## Dataset

- Dataset: **Titanic: Machine Learning from Disaster** (Kaggle).
- The file you uploaded (`train.csv`) is the standard Kaggle Titanic training file.
- It must be placed in the `data/` folder with this exact path:

  ```text
  data/train.csv
  ```

## How to Run

1. **(Optional) Create and activate a virtual environment**:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Add the dataset**:

   - Ensure `train.csv` is inside the `data/` directory:

     ```text
     data/train.csv
     ```

4. **Run the script**:

   ```bash
   python titanic_feature_engineering.py
   ```

   The script will:
   - Print basic info and missing values.
   - Perform feature engineering:
     - Extract `Title` from `Name`
     - Create `FamilySize`, `IsAlone`, `CabinKnown`
   - Handle missing values with imputers.
   - One-hot encode all categorical features.
   - Train a `RandomForestClassifier`.
   - Print accuracy and a full classification report.
   - Save a feature-importance plot to `outputs/feature_importances.png`.

## Main Techniques Used

- **Feature Engineering Examples**
  - `Title` from `Name` (e.g., Mr, Mrs, Miss, Master, Rare).
  - `FamilySize = SibSp + Parch + 1`
  - `IsAlone` = 1 if `FamilySize == 1`, else 0.
  - `CabinKnown` = 1 if the passenger has a cabin value, else 0.

- **Handling Missing Values**
  - Numeric columns (e.g., `Age`, `Fare`) → imputed using **median**.
  - Categorical columns (e.g., `Embarked`) → imputed using **most frequent value**.

- **Encoding Categorical Variables**
  - `OneHotEncoder` from scikit-learn with `handle_unknown="ignore"`.

- **Modeling**
  - `RandomForestClassifier(n_estimators=300, random_state=42)`
  - Evaluation with:
    - `accuracy_score`
    - `classification_report` (precision, recall, F1-score)

- **Feature Importance**
  - Extracted from the Random Forest model using `.feature_importances_`.
  - Visualized using a horizontal bar chart and saved as `outputs/feature_importances.png`.

## Short Conceptual Answers (for your report / viva)

**1. What is feature engineering, and why is it important?**  
Feature engineering is the process of transforming raw data into features that better represent the underlying patterns in the data. Good features make it easier for machine learning models to learn, leading to **better accuracy and generalization**.

**2. How does feature engineering improve model performance on the Titanic dataset?**  
By adding features like `Title`, `FamilySize`, `IsAlone`, and `CabinKnown`, we capture important information about a passenger's **social status**, **family group**, and **ticket class/wealth**, all of which strongly relate to survival probability. This gives the model more signal and less noise.

**3. How did you handle missing values?**  
- For numerical variables (e.g., Age, Fare), we used **median imputation**.  
- For categorical variables (e.g., Embarked), we used **most frequent category** imputation.  
These strategies are simple but effective and avoid dropping many rows.

**4. Why use One-Hot Encoding for categorical variables?**  
Many models (like Random Forests) cannot work directly with string categories. One-Hot Encoding converts each category into a separate binary column, allowing the model to learn separate effects for each category without assuming any ordering.

**5. Why choose RandomForestClassifier for this task?**  
Random Forest:
- Handles both numeric and categorical (after encoding) data well.
- Is robust to outliers and noise.
- Automatically captures **non-linear relationships** and **interactions** between features.
- Provides **feature importance**, which is great for model interpretability.

## How to Upload to GitHub

1. Create a new GitHub repository (for example, `data-analytics-task-08-titanic`).
2. Add all files from this folder:
   - `titanic_feature_engineering.py`
   - `requirements.txt`
   - `README.md`
   - `data/` (with `train.csv` in your local copy – you may choose to *not* push it if required by rules)
   - `outputs/` (after running the script so it contains `feature_importances.png`)
3. Commit and push:

   ```bash
   git init
   git add .
   git commit -m "Add solution for Data Analytics Internship Task 08 (Titanic)"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

Then you can submit your **GitHub repository link** for the internship task.
