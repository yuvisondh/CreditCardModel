import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, average_precision_score,precision_recall_curve
from tensorflow import keras

# Load the dataset
df = pd.read_csv('creditcard.csv')
# Check for missing values
print(df.isnull().sum())

print(df.head())

print(df.shape)
print(df['Class'].value_counts())


# Split the data into training and testing sets

X = df.drop('Class', axis=1) # Features (all columns except 'Class')
y = df['Class'] # Target variable (the 'Class' column) or Label 


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 20% saved for testing, 80% for training
    random_state=42,    
    stratify=y       
)   

# Scale the features
scaler = StandardScaler()

X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']]  = scaler.transform(X_test[['Time', 'Amount']])



