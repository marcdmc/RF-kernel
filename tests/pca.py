"""This example performs PCA on a dataset using both regular PCA and kernel PCA."""
#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
import time
import matplotlib.pyplot as plt
#%%
import sys
sys.path.insert(0, '../src')
import RandomForestKernel as rfk


df = pd.read_csv('datasets/Data_for_UCI_named.csv')
df = df[0:2000]
df = pd.DataFrame.to_numpy(df)

# Separate label
x = df[:, 0:12]
y = df[:, 13]
# Split into train-test
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.2)

r = rfk.RandomForestKernel(x_training_data, y_training_data)
K_train = r.K_train
K_test  = r.transform(x_test_data)

# Our kernel PCA
#%%
pca = KernelPCA(kernel = 'precomputed')
res = pca.fit(K_train).transform(K_test)

colors = [tag == 'unstable' for tag in y_test_data]
plt.scatter(res[:,0],res[:,1], c=colors)
plt.show()

# Regular PCA
# %%
pca = PCA()
res = pca.fit(x_training_data).transform(x_test_data)

colors = [tag == 'unstable' for tag in y_test_data]
plt.scatter(res[:,0],res[:,1], c=colors)
plt.show()
# %%
