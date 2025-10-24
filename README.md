# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preprocessing
2. Train–Test Split
3. Model Training and Prediction
4. Model Evaluation
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: shivasri
RegisterNumber:  212224220098
*/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # keep output tidy; remove if you want warnings

def load_sample_data():
    # Example dataset — replace with your CSV read if you have a file
    return pd.DataFrame({
        'Age': [25, 30, 45, 35, 40, 50, 28, 32, 41, 36],
        'Salary': [30000, 40000, 80000, 50000, 60000, 90000, 32000, 42000, 70000, 52000],
        'YearsAtCompany': [1, 3, 10, 5, 7, 12, 2, 4, 9, 6],
        'Department': ['Sales', 'HR', 'Tech', 'Tech', 'HR', 'Sales', 'Sales', 'Tech', 'HR', 'Sales'],
        'Churn': [1, 0, 0, 0, 0, 1, 1, 0, 0, 0]  # 1=Left, 0=Stayed
    })

def load_data_from_csv(path):
    # If you have your own CSV with a 'Churn' column:
    df = pd.read_csv(path)
    return df

def prepare_and_train(df, target_col='Churn', random_state=42):
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Detect numerical and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build preprocessing: scale numeric, one-hot encode categorical (ignore unknowns)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
        ],
        remainder='drop'  # drop any other columns
    )

    # Pipeline: preprocessing -> classifier
    pipeline = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', DecisionTreeClassifier(criterion='entropy', random_state=random_state))
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y if len(np.unique(y))>1 else None)

    # Fit
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    return pipeline, preprocessor, num_cols, cat_cols, (X_test, y_test, y_pred)

def plot_decision_tree(pipeline, preprocessor, feature_names=None, class_names=None):
    # Extract trained DecisionTreeClassifier from the pipeline
    clf = pipeline.named_steps['clf']
    # Build feature names after preprocessing
    # If feature_names provided, use them; else try to reconstruct from preprocessor
    if feature_names is None:
        # numeric names
        num_names = []
        cat_names = []
        # If ColumnTransformer was used
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                num_names = list(cols)
            if name == 'cat':
                # get categories from the OneHotEncoder inside transformer
                ohe = trans
                # ohe.categories_ is list of arrays
                for col, cats in zip(cols, ohe.categories_):
                    cat_names.extend([f"{col}__{c}" for c in cats])
        feature_names = num_names + cat_names

    plt.figure(figsize=(14,8))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, max_depth=4)
    plt.title("Decision Tree (truncated view)")
    plt.show()

def predict_new(pipeline, new_row: dict):
    """
    new_row: dict of column:value matching original feature columns (before encoding),
             e.g. {'Age':29,'Salary':45000,'YearsAtCompany':3,'Department':'Sales'}
    """
    df_new = pd.DataFrame([new_row])
    pred = pipeline.predict(df_new)
    prob = pipeline.predict_proba(df_new) if hasattr(pipeline.named_steps['clf'], "predict_proba") else None
    print("Input:", df_new.to_dict(orient='records')[0])
    print("Predicted Churn:", int(pred[0]), "(1=Left, 0=Stayed)")
    if prob is not None:
        print("Prediction probabilities:", prob[0])

if __name__ == "__main__":
    # ---------- Choose data source ----------
    use_csv = False   # <-- change to True if you want to load from CSV
    csv_path = "employee_churn.csv"  # put your CSV path here

    if use_csv:
        try:
            df = load_data_from_csv(csv_path)
            print(f"Loaded CSV with shape: {df.shape}")
        except Exception as e:
            print("Failed to load CSV:", e)
            print("Falling back to sample dataset.")
            df = load_sample_data()
    else:
        df = load_sample_data()

    # Basic checks to avoid common errors
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' not found in data. Rename your target column to 'Churn' or change the code accordingly.")

    # Train model
    pipeline, preprocessor, num_cols, cat_cols, test_info = prepare_and_train(df)

    # Plot tree (optional). If categorical had many columns, feature names may be long.
    try:
        plot_decision_tree(pipeline, preprocessor, class_names=['Stayed','Left'])
    except Exception as e:
        print("Plotting failed (maybe too many features). Error:", e)

    # Example predict for a new employee (modify values as needed)
    new_employee = {'Age': 29, 'Salary': 45000, 'YearsAtCompany': 3, 'Department': 'Sales'}
    predict_new(pipeline, new_employee)

    print("\nNumeric columns detected:", num_cols)
    print("Categorical columns detected:", cat_cols)
```

## Output:
<img width="1607" height="939" alt="image" src="https://github.com/user-attachments/assets/42a6d55b-ef4b-4f4f-8376-dee5d497ec91" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
