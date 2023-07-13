#!/usr/bin/env python3.8

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

free_array = np.load('../Qp/psd_new/free_array.npy')
contact_array = np.load('../Qp/psd_new/contact_3_array.npy')
operator_array = np.load('../Qp/psd_new/operator_2_array.npy')

data_labels = np.hstack((0*np.ones(len(free_array)), 1*np.ones(len(contact_array)), 2*np.ones(len(operator_array))))
data = np.vstack((free_array, contact_array, operator_array))
normalized_data = np.divide(data, np.max(data, axis=0))

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_data, data_labels, test_size=0.2, random_state=42)

# Step 3: Create and train the classifier
classifier = KNeighborsClassifier(n_neighbors=3)  # You can use any classifier of your choice
classifier.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_pred = classifier.predict(X_test)

# Step 5: Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# save
knnPickle = open('../knc/knnpickle_last', 'wb')
# source, destination
pickle.dump(classifier, knnPickle)
# close the file
knnPickle.close()
