import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

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

print("Example of RFK with ",os.cpu_count()," cores")

r = rfk.RandomForestKernel(x_training_data, y_training_data)
print(r.K_train)