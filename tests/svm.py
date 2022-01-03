import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, '../src')
import RandomForestKernel as rfk

df = pd.read_csv('datasets/Data_for_UCI_named.csv')
# Select only first 2k rows
df = df[0:2000]
df = pd.DataFrame.to_numpy(df)
# Separate label
x = df[:, 0:12]
y = df[:, 13]
# Split into train-test
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.2)

r = rfk.RandomForestKernel(x_training_data, y_training_data, x_test_data)

# Train a SVC with the kernelized matrix
clf = svm.SVC(kernel = 'precomputed')
clf.fit(r.K_train, y_training_data)
y_pred_test = clf.predict(r.K_test)

print(confusion_matrix(y_test_data, y_pred_test))