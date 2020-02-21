import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

### Setup ###
data = pd.read_csv('creditcard.csv')
data = data.sample(frac = 1, random_state = 1)
"""
data.hist(figsize = (15, 15))
plt.show()
"""
print("Data Shape")
print(data.shape)
print()

#  Determine fraud cases
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_frac = len(fraud)/float(len(valid))
print("Outlier Fraction Percentage")
print(outlier_frac*100)  # Approx % fraud cases
print()

#  Create correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = 0.8, square = True)
plt.show()

#  Get all columns from dataframe
columns = data.columns.tolist()

#  Filter columns to remove data unwanted
columns = [c for c in columns if c not in ['Class']]

#  Store variable predicted
target = 'Class'
X = data[columns]
Y = data[target]


### Anomaly Detection ###
state = 1  # define random state
classifiers = {  # define outlier detection method
    "Isolation Forest": IsolationForest(max_samples=len(X), contamination=outlier_frac, random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=outlier_frac)
}


# Begin Fitting Model #
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    if (clf_name == "Local Outlier Factor"):
        y_predict = clf.fit_predict(X)
        scores_predict = clf.negative_outlier_factor_
    else:
        y_predict = clf.fit(X).predict(X)

    # Reshape predictions 0 -> valid, 1 -> fraud
    y_predict[y_predict == 1] = 0
    y_predict[y_predict == -1] = 1
    n_errors = (y_predict != Y).sum()

    # Run Classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_predict))
    print(classification_report(Y, y_predict))
