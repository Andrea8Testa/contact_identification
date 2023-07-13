#!/usr/bin/env python3.8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from joblib import dump

free_array = np.load('../Qp/psd/free_ext_array.npy')
contact_array = np.load('../Qp/psd/contact_ext_array.npy')
operator_array = np.load('../Qp/psd/operator_ext_array.npy')

data = np.vstack((free_array, contact_array, operator_array))
row_sums = np.sum(data, axis=1)
normalized_data = np.divide(data, row_sums[:, np.newaxis])
data_with_energy = np.hstack((normalized_data, (row_sums.reshape(-1, 1))/max(row_sums)))
# Create an instance of Gaussian Mixture Models
num_clusters = 15
gmm = GaussianMixture(n_components=num_clusters, random_state=0, max_iter=1000, n_init=10)

# Fit the GMM to your data
gmm.fit(data_with_energy)

# Obtain the probabilistic cluster assignments for each point
proba_matrix = gmm.predict_proba(data_with_energy)
labels = gmm.predict(data_with_energy)
'''
# Iterate over each point and print the cluster probabilities
for i, proba_row in enumerate(proba_matrix):
    point = data[i]
    cluster_probs = zip(range(num_clusters), proba_row)
    print(f"Point {i}:")
    for cluster, prob in cluster_probs:
        print(f"    Cluster {cluster}: Probability = {prob}")
'''
# Divide the label matrix into three submatrices
submatrix_lengths = [free_array.shape[0], contact_array.shape[0], operator_array.shape[0]]
submatrices = np.split(labels, np.cumsum(submatrix_lengths)[:-1])

# Count the occurrences of labels in each submatrix
label_counts = []
for submatrix in submatrices:
    unique_labels, counts = np.unique(submatrix, return_counts=True)
    label_counts.append(dict(zip(unique_labels, counts)))

# Plot the pie chart for each submatrix
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
labels_list = list(range(num_clusters))  # Labels

for i, ax in enumerate(axs):
    counts = label_counts[i]
    values = [counts.get(label, 0) for label in labels_list]
    ax.pie(values, labels=labels_list, autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Submatrix {i+1}")

plt.tight_layout()
plt.show()

# Get the unique labels
labels_unique = np.unique(labels)

# Initialize arrays to store the counts for each label and submatrix
counts_submatrix1 = np.zeros_like(labels_unique)
counts_submatrix2 = np.zeros_like(labels_unique)
counts_submatrix3 = np.zeros_like(labels_unique)

# Collect the counts for each label and submatrix
for label, count in label_counts[0].items():
    counts_submatrix1[label] = count

for label, count in label_counts[1].items():
    counts_submatrix2[label] = count

for label, count in label_counts[2].items():
    counts_submatrix3[label] = count

# Plot the stacked bar chart
x = np.arange(len(labels_unique))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x, counts_submatrix1, width, label='Submatrix 1')
ax.bar(x, counts_submatrix2, width, bottom=counts_submatrix1, label='Submatrix 2')
ax.bar(x, counts_submatrix3, width, bottom=counts_submatrix1+counts_submatrix2, label='Submatrix 3')

ax.set_xlabel('Label')
ax.set_ylabel('Occurrences')
ax.set_title('Label Distribution in Submatrices')
ax.set_xticks(x)
ax.set_xticklabels(labels_unique)
ax.legend()

plt.tight_layout()
plt.show()

dump(gmm, '../Qp/models/gmm_model_ext_15.joblib')
