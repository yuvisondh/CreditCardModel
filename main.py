import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, average_precision_score,precision_recall_curve
from tensorflow import keras

from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('creditcard.csv')
# Check for missing values
#print(df.isnull().sum())

print(df.head())

print(df.shape)
#print(df['Class'].value_counts())


# Split the data into training and testing sets

X = df.drop('Class', axis=1) # Features (all columns except 'Class')
y = df['Class'].astype(int).values # Target variable (the 'Class' column) or Label 


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

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Build the neural network model    
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_resampled.shape[1],)),
    keras.layers.dropout(0.3),  # Dropout layer to prevent overfitting

    keras.layers.Dense(32, activation='relu'),
    keras.layers.dropout(0.3),

    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification 0 or 1
])

model.compile(
    optimizer='adam', # used adaptive learning rate optimization algorithm that is efficient and widely used for training deep learning models.
    loss='binary_crossentropy', # used for binary classification problems, where the target variable has two classes (0 and 1). It measures the difference between the predicted probabilities and the actual class labels.
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]

)



