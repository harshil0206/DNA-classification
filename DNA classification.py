import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn import model_selection
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10000)

# import the uci molecular biology (promoter gene sequences) dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"
names = ["Class","id","Sequence"]
data = pd.read_csv(url, names = names)
# print(data.iloc[:5])

# build our dataset using a custom pandas dataframe
# each column in a dataframe is called a series

classes = data.loc[:,"Class"]
# print(classes[:5])

# generate a list of DNA sequences
sequences = list(data.loc[:, "Sequence"])
dataset = {}

# loop throught the sequences and split into individual nucleotides
for i, seq in enumerate(sequences):

    # split into nucleaotides, remove tab characters
    nucleotides = list(seq)
    nucleotides = [x for x in nucleotides if x != "\t"]

    # append the class assignment
    nucleotides.append(classes[i])

    #add to the dataset
    dataset[i] = nucleotides

# print(dataset[0])

# turn dataset into pandas dataframe
dframe = pd.DataFrame(dataset)
# print(dframe)

# transpose the dataframe
df = dframe.transpose()
# print(df.iloc[:5])

# rename the lasta column to class
df.rename(columns = {57: "Class"}, inplace = True)
# print(df.iloc[:5])

# print(df.describe())

# record value count for each sequence
series = []
for name  in df.columns:
    series.append(df[name].value_counts())

info = pd.DataFrame(series)
details = info.transpose()
# print(details)

# switch to numerical data using pd.get_dummies() function
numerical_df = pd.get_dummies(df)
# print(numerical_df.iloc[:5])

# remove one of the class columns and rename to simply "Class"
df = numerical_df.drop(columns=["Class_-"])

df.rename(columns = {"Class_+" : "Class"}, inplace=True)
# print(df.iloc[:5])

# import the algorithms

# define the scoring method
scoring = "accuracy"

# define the models to train
names = ["K Nearest Neighbors", "Gaussian Process",
         "Decision Tree", "Random Forest",
         "Neural Net", "AdaBoost",
         "Naive Bayes", "SVM Linear",
         "SVM RBF", "SVM Sigmoid"]

classifiers = [KNeighborsClassifier(n_neighbors=3),
               GaussianProcessClassifier(1.0 * RBF(1.0)),
               DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               MLPClassifier(alpha=1),
               AdaBoostClassifier(),
               GaussianNB(),
               SVC(kernel="linear"),
               SVC(kernel="rbf"),
               SVC(kernel="sigmoid")]

models = zip(names,classifiers)

# evaluate each model in turn
results = []
names = []

# Create X and y datasets for training
X = np.array(df.drop(["Class"], 1))
y = np.array(df["Class"])

# define a seed for reproducibility
seed = 1

# split the data into training and testing datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=seed)

# for name, model in models:
#     kfold = model_selection.KFold(n_splits = 10, random_state = seed)
#     cv_results = model_selection.cross_val_score\
#         (model, X_train, y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "{0}: {1} ({2})".format(name,cv_results.mean(), cv_results.std())
#     print(msg)

# test the algorithms on the validation dataset

for name, model in models:
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

