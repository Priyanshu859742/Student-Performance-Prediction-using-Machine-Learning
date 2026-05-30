import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("="*50)
    print(" STUDENT PERFORMANCE PREDICTION ML PROJECT")
    print("="*50)
    
    # 1. Load Data
    data_path = os.path.join('data', 'student_performance.csv')
    if not os.path.exists(data_path):
        print("Data not found. Generating dataset first...")
        from src.generate_data import generate_student_data
        df = generate_student_data()
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    
    print(f"\n[INFO] Loaded dataset with {len(df)} records.")
    
    # Feature selection
    X = df[['Study_Hours', 'Attendance_Percentage', 'Previous_Marks', 'Assignments_Completed']]
    
    print("\n" + "="*50)
    print(" TASK 1: REGRESSION (Predicting Final Score)")
    print("="*50)
    y_reg = df['Final_Score']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    reg_model = LinearRegression()
    reg_model.fit(X_train_r, y_train_r)
    y_pred_r = reg_model.predict(X_test_r)
    
    print(f"Linear Regression Results:")
    print(f" - Mean Absolute Error (MAE): {mean_absolute_error(y_test_r, y_pred_r):.2f}")
    print(f" - Mean Squared Error (MSE):  {mean_squared_error(y_test_r, y_pred_r):.2f}")
    print(f" - R-squared (R2) Score:      {r2_score(y_test_r, y_pred_r):.2f}")
    
    print("\n" + "="*50)
    print(" TASK 2: CLASSIFICATION (Predicting Pass/Fail)")
    print("="*50)
    y_clf = df['Pass_Fail']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train_c, y_train_c)
    y_pred_log = log_reg.predict(X_test_c)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test_c, y_pred_log)*100:.2f}%\n")
    
    # Decision Tree
    dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_clf.fit(X_train_c, y_train_c)
    y_pred_dt = dt_clf.predict(X_test_c)
    print(f"Decision Tree Accuracy:       {accuracy_score(y_test_c, y_pred_dt)*100:.2f}%")
    print("\nDecision Tree Classification Report:")
    print(classification_report(y_test_c, y_pred_dt))
    
    print("="*50)
    print(" PROJECT EXECUTION COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()
