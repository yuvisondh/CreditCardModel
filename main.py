import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, average_precision_score,precision_recall_curve
from tensorflow import keras

# Load the dataset
data = pd.read_csv('creditcard.csv')
# Check for missing values
print(data.isnull().sum())

