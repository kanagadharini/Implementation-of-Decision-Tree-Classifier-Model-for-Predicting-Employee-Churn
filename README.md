# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Create a sample dataset (since 'employee_churn.csv' was not found)
data = pd.DataFrame({
    'Age': [25, 45, 30, 50, 28, 40, 35],
    'Salary': [40000, 80000, 50000, 90000, 42000, 75000, 62000],
    'Department': ['Sales', 'HR', 'IT', 'Finance', 'Sales', 'IT', 'HR'],
    'Tenure': [2, 10, 4, 12, 3, 8, 6],
    'Churn': [1, 0, 1, 0, 1, 0, 0]
})

# Step 2: Preprocess the data
data = pd.get_dummies(data, drop_first=True)

# Step 3: Define features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Initialize and train the Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

## Output:
![alt text](<Screenshot 2025-10-07 013128.png>)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
