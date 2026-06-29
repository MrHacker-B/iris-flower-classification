import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the Iris dataset
data = pd.read_csv('iris.csv')

# Display the first few rows of the dataset
print(data.head())

# Display the shape of the dataset
print("Shape of the dataset: {}".format(data.shape))

# coloumns of the dataset
print("Columns in the dataset: {}".format(data.columns))

# dataset information
print("Dataset information:")
print(data.info())

# Display summary statistics of the dataset
print("Summary statistics of the dataset:")
print(data.describe())

# Check for missing values in the dataset
print("Missing values in the dataset:")
print(data.isnull().sum())

# Count Plot
sns.countplot(x='Species', data=data)
plt.title('Count of each species in the Iris dataset')  # type: ignore
plt.show()
sns.pairplot(data, hue='Species')
plt.show()

# Correlation Heatmap
numeric_data = data.drop('Species', axis=1)

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Remove Id column
data = data.drop("Id", axis=1)

# Convert Species text into numbers
encoder = LabelEncoder()
data["Species"] = encoder.fit_transform(data["Species"])

# Features (X) and Target (y)
X = data.drop("Species", axis=1)
y = data["Species"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Create the Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Make predictions on the test set
y_pred = model.predict(X_test)

print("\nPredictions on the test set:")
print(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy*100, "%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Predicting the species of a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example sample
predicted_species = model.predict(sample)

flower = encoder.inverse_transform(predicted_species)
print("\nPredicted species for the sample {}: {}".format(sample, flower[0]))


joblib.dump(model, 'iris_model.pkl')
print("\nModel saved successfully as 'iris_model.pkl'.")