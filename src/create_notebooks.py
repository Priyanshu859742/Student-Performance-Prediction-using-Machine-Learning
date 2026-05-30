import nbformat as nbf
import os

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell("# Exploratory Data Analysis and Preprocessing"))
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
data_path = os.path.join('..', 'data', 'student_performance.csv')
df = pd.read_csv(data_path)
df.head()"""))
    cells.append(nbf.v4.new_markdown_cell("## Data Understanding"))
    cells.append(nbf.v4.new_code_cell("""df.info()\ndf.describe()"""))
    cells.append(nbf.v4.new_markdown_cell("## Correlation Analysis"))
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()"""))
    cells.append(nbf.v4.new_markdown_cell("## Visualizing Targets"))
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Final_Score'], bins=20, kde=True)
plt.title('Distribution of Final Score')

plt.subplot(1, 2, 2)
sns.countplot(x='Pass_Fail', data=df)
plt.title('Pass vs Fail Count')
plt.show()"""))
    nb['cells'] = cells
    with open('../notebooks/01_EDA_and_Preprocessing.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_regression_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell("# Regression: Predicting Final Score"))
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_path = os.path.join('..', 'data', 'student_performance.csv')
df = pd.read_csv(data_path)
"""))
    cells.append(nbf.v4.new_markdown_cell("## Train Test Split"))
    cells.append(nbf.v4.new_code_cell("""X = df[['Study_Hours', 'Attendance_Percentage', 'Previous_Marks', 'Assignments_Completed']]
y = df['Final_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)"""))
    cells.append(nbf.v4.new_markdown_cell("## Model Training"))
    cells.append(nbf.v4.new_code_cell("""model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)"""))
    cells.append(nbf.v4.new_markdown_cell("## Evaluation"))
    cells.append(nbf.v4.new_code_cell("""mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")"""))
    cells.append(nbf.v4.new_markdown_cell("## Visualization of Actual vs Predicted"))
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Final Score')
plt.ylabel('Predicted Final Score')
plt.title('Actual vs Predicted Final Score')
plt.show()"""))
    nb['cells'] = cells
    with open('../notebooks/02_Regression_Final_Score.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_classification_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell("# Classification: Predicting Pass/Fail"))
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data_path = os.path.join('..', 'data', 'student_performance.csv')
df = pd.read_csv(data_path)
"""))
    cells.append(nbf.v4.new_markdown_cell("## Train Test Split"))
    cells.append(nbf.v4.new_code_cell("""X = df[['Study_Hours', 'Attendance_Percentage', 'Previous_Marks', 'Assignments_Completed']]
y = df['Pass_Fail']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)"""))
    cells.append(nbf.v4.new_markdown_cell("## Logistic Regression"))
    cells.append(nbf.v4.new_code_cell("""log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_log))"""))
    cells.append(nbf.v4.new_markdown_cell("## Decision Tree Classifier"))
    cells.append(nbf.v4.new_code_cell("""dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_dt))"""))
    cells.append(nbf.v4.new_markdown_cell("## Confusion Matrix (Decision Tree)"))
    cells.append(nbf.v4.new_code_cell("""cm = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision Tree')
plt.show()"""))
    nb['cells'] = cells
    with open('../notebooks/03_Classification_Pass_Fail.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    print("Creating notebooks...")
    os.makedirs('../notebooks', exist_ok=True)
    create_eda_notebook()
    create_regression_notebook()
    create_classification_notebook()
    print("Notebooks created successfully!")
