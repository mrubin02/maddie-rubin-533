import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# importing or loading the dataset
dataset = pd.read_csv('hw4.csv')

# distributing the dataset into two components X and Y
X = dataset[['ECRPUS 1Y Index','SPXSFRCS Index','FDTRFTRL Index']]
y = dataset['success']

# Splitting the X and Y into the
# Training set and Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# performing preprocessing part
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA function on training
# and testing set of X component

pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_




# Fitting Logistic Regression To the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the test set result using
# predict function under LogisticRegression
y_pred = classifier.predict(X_test)

print(y_pred)

# making confusion matrix between
# test set of Y and predicted value.

cm = confusion_matrix(y_test, y_pred)
print(cm)


Accuracy = metrics.accuracy_score(y_test, y_pred)
print(Accuracy)