## 1/2 - GET DATA

import pandas as pd

path = "XXXXXXXXX"
data_raw = pd.read_csv(path)
data_raw.head()

## 3a - INSIGHTS (data)

data_raw.info()           ## Check for column mismatch

# categorical data: how many types per category?
print(data_raw["XXX"].value_counts())     # category 1 (x5)
print(data_raw["XXX"].value_counts())     # category 2 (x5)

data_raw.describe()                       # desc stats. Not massively useful here, but maxima and minima are always good to know

## 3b - INSIGHTS (graphical)

%matplotlib inline                  # graphs currently for kaggle/jupyter
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

correlates_matrix = data_raw.corr()
correlates_matrix["XXX"].sort_values(ascending=False)
num_attributes = ["XXX","XXX","XXX","XXX","XXX","XXX","XXX"]       # categories de-identified
scatter_matrix(data_raw[num_attributes], figsize=(20,20), alpha=0.1)  # self vs self will be histograms, vs non self will be scatter graphs
plt.show()   

## 3b - continued...

# plot x as XXX, y as XXX, with overlaid heatmap for XXX
data_raw.plot(kind="scatter",x="XXX",y="XXX", alpha=0.1, figsize=(15,12), c="XXX", cmap=plt.get_cmap("jet"), colorbar=True)
plt.show()

## 3c - DIMENSIONALITY REDUCTION

# calculate mean for XXX
data_raw["XXX"]=(data_raw["XXX"] + data_raw["XXX"] + data_raw["XXX"]) / 3
data_raw.head()  # check it worked

## 3d - MORE INSIGHTS 

# as in 3b, but heatmap with ired_mean
data_raw.plot(kind="scatter",x="XXX",y="XXX", alpha=0.1, figsize=(15,12), c="XXX", cmap=plt.get_cmap("jet"), colorbar=True)
plt.show()

# pattern seems to approximate that in 3b, indicating reduced dimension is probably appropriate.
# There is less variation in XXX, indicating XXX could perhaps be more sensitive for the application.


## 4 - Data splitting & prep

from sklearn.model_selection import train_test_split 

# reduce dimensions of XXX to use only calibrated mean of XXX0-2
data_reduced = data_raw.drop(["XXX","XXX","XXX","XXX","XXX","XXX"],axis=1)
# remove XXX as I'm not yet interested in looking at intra-device variation
data_reduced = data_reduced.drop(["XXX"],axis=1)

train_set, test_set = train_test_split(data_reduced, test_size=0.2,random_state=25)
print("Training set size: ", len(train_set), " . Test set size: ", len(test_set))

# split labels
train_set_preclean = train_set.drop("XXX",axis=1)
train_set_labels = train_set["XXX"].copy()

# binarise XXX empty/not empty
train_set_labels_binarised = (train_set_labels == 0)
#train_set_labels_binarised.head()

## 5 - Data cleaning

from sklearn.preprocessing import LabelBinarizer

# check for empty cells
sample_incomplete_rows = train_set_preclean[train_set_preclean.isnull().any(axis=1)].head()
sample_incomplete_rows

train_set_numOnly = train_set_preclean.drop(["XXX"],axis=1)     # drop nonnumerical data
train_set_postclean = train_set_numOnly

# encode categorical data
train_set_cat = train_set_preclean[["XXX"]]
print(train_set_cat.head(10))

lb = LabelBinarizer()
train_set_cat_onehot = lb.fit_transform(train_set_cat)
lb_df = pd.DataFrame(train_set_cat_onehot, index=train_set_cat.index)  ## Note: tidy up references to past model using onehot encoder


data_cleaned = pd.concat([train_set_postclean,lb_df],axis=1)
data_cleaned.head(5)

#data_cleaned[data_cleaned.isnull().any(axis=1)].head()  # double check no empty fields after concat

## Note: automate with pipeline

## 6 - evaluate model
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score

# put into numpty arrays
X = np.array(data_cleaned)
y = np.array(train_set_labels_binarised)

print("Actual y labels: ", y)

lr = LogisticRegression(random_state=25)
lr.fit(X, y)
preds = lr.predict(X)
print("Logistic Regression prediction: ", preds)
print("Precision score: ",precision_score(y,preds))
print("Recall score: ",recall_score(y,preds))
print("..............")

sgd_clf = SGDClassifier(random_state=25)
sgd_clf.fit(X,y)
preds = sgd_clf.predict(X)
print(preds)
print("Stochastic Gradient Descent prediction: ", preds)
print("Precision score: ",precision_score(y,preds))
print("Recall score: ",recall_score(y,preds))
print("..............")



