#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn import model_selection
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score

#load the csv file
iris = datasets.load_iris()
X = iris.data[:, :]  # we only take the first two features.
y = iris.target

#Initialize Gaussian Naive Bayes
NB_clf = GaussianNB()

# One-third of data as a part of test set
validation_size = 0.33

seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
scoring = 'accuracy'

#Naive Bayes Classifier
#Fitting the training set
NB_clf.fit(X_train, Y_train) 

#Predicting for the Test Set
pred_clf = NB_clf.predict(X_validation)

#Prediction Probability
prob_pos_clf = NB_clf.predict_proba(X_validation)[:, 1]

#Model Performance
#setting performance parameters
kfold = model_selection.KFold(n_splits=10, random_state=seed)

#calling the cross validation function
cv_results = model_selection.cross_val_score(GaussianNB(), X_train, Y_train, cv=kfold, scoring=scoring)

#displaying the mean and standard deviation of the prediction
print()
print("NB Classifier")
msg = "%s: %f (%f)" % ('NB accuracy', cv_results.mean(), cv_results.std())
print(msg)


#minimun distace Classifier
MDC_clf = NearestCentroid()


#fitting the training set
MDC_clf.fit(X_train, Y_train)

#predicting for test set
pred_MDC_clf = MDC_clf.predict(X_validation)

#Model performance
kfold = model_selection.KFold(n_splits=10, random_state=seed)

cv_results = model_selection.cross_val_score(NearestCentroid(), X_train, Y_train, cv=kfold, scoring=scoring)
print()
print("MDC Classifier")

msg = "%s: %f (%f)" % ('NB accuracy', cv_results.mean(), cv_results.std())
print(msg)
print()


