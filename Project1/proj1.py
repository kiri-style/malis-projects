import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data
df = pd.read_csv('/Users/user/Documents/EURECOM/S7/Malis/Projects/malis-projects/Project1/iris.csv')

# Encoding the classes
le = LabelEncoder()
df['class'] = le.fit_transform(df['variety'])

# Separating features (X) and labels (y)
X = df.drop(['variety', 'class'], axis=1)
y = df['class']

# Task 1:

# Filtering to use only two classes (Setosa and Versicolor)
df_binary = df[df['class'] != 2]  # Keep only classes 0 and 1
X_binary = df_binary.drop(['variety', 'class'], axis=1)
y_binary = df_binary['class']

# Splitting into training and test sets
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)

# Training a linear regression model for binary classification
model_bin = LinearRegression()
model_bin.fit(X_train_bin, y_train_bin)

# Prediction
y_pred_bin = model_bin.predict(X_test_bin)

# Converting predictions to binary classes (decision threshold: 0.5)
y_pred_bin_class = (y_pred_bin > 0.5).astype(int)

# Calculating accuracy
accuracy_bin = accuracy_score(y_test_bin, y_pred_bin_class)
print(f"Binary model accuracy: {accuracy_bin * 100:.2f}%")

# Visualization:
# Comparison plot of predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test_bin)), y_test_bin, color='blue', label='Actual values', alpha=0.6)
plt.scatter(range(len(y_pred_bin_class)), y_pred_bin_class, color='red', marker='x', label='Predictions', alpha=0.6)
plt.title("Comparison between predictions and actual values (binary)")
plt.xlabel("Examples")
plt.ylabel("Class")
plt.legend()
plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_test_bin, y_pred_bin_class)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("Confusion matrix for binary model")
plt.xlabel("Predictions")
plt.ylabel("Actual values")
plt.show()

# Task 2:

# Splitting into training and test sets
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X, y, test_size=0.3, random_state=42)

# Training three regression models (One-vs-Rest)
models = {}
for class_label in np.unique(y):
    # Create a model for each class
    y_train_binary = (y_train_multi == class_label).astype(int)
    model = LinearRegression()
    model.fit(X_train_multi, y_train_binary)
    models[class_label] = model

# Multi-class prediction
y_pred_multi = np.zeros((X_test_multi.shape[0], len(models)))
for class_label, model in models.items():
    y_pred_multi[:, class_label] = model.predict(X_test_multi)

# Taking the class with the highest predicted value as the final decision
y_pred_multi_class = np.argmax(y_pred_multi, axis=1)

# Calculating accuracy for the multi-class model
accuracy_multi = accuracy_score(y_test_multi, y_pred_multi_class)
print(f"Multi-class model accuracy: {accuracy_multi * 100:.2f}%")

# Visualization:

# Confusion matrix for the multi-class model
conf_matrix_multi = confusion_matrix(y_test_multi, y_pred_multi_class)
sns.heatmap(conf_matrix_multi, annot=True, cmap="Blues", fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predictions")
plt.ylabel("Actual classes")
plt.title("Confusion matrix for multi-class model")
plt.show()