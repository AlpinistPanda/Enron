#!/usr/bin/python

import sys
import pickle
sys.path.append("./tools/")

from tester import dump_classifier_and_data

import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
# from tester import dump_classifier_and_data


from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tester import dump_classifier_and_data
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn import preprocessing

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
# features_list = ['poi','salary'] # You will need to use more features

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
data_dict.pop('TOTAL', None)

missingValues = {}
for person in data_dict:
    n = 0
    for value in data_dict[person].items():
        if value[1] == 'NaN':
            n += 1
    missingValues[person] = n

# Remove unnecessary ones

data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)
data_dict.pop("LOCKHART EUGENE E", None)
data_dict.pop('WHALEY DAVID A', None)
data_dict.pop('WROBEL BRUCE', None)
data_dict.pop('GRAMM WENDY L', None)


missingData = {}
for person in data_dict:
    for key, value in data_dict[person].items():
        if value == "NaN":
            if key in missingData:
                missingData[key] += 1
            else:
                missingData[key] = 1


fieldsToRemove = [
    "bonus",
    "deferral_payments",
    "deferred_income",
    "director_fees",
    "exercised_stock_options",
    "expenses",
    "loan_advances",
    "long_term_incentive",
    "other",
    "restricted_stock",
    "restricted_stock_deferred",
    "salary",
    "total_payments",
    "total_stock_value"
]


# Give 0 for missing data

for field in fieldsToRemove:
    for person in data_dict:
        if data_dict[person][field] == "NaN":
            data_dict[person][field] = 0


# Extract features and labels from dataset for local testing
dataPD = pd.DataFrame(data_dict)


# Transpose
dataPD2 = pd.DataFrame()
dataPD2 = dataPD.transpose()


feature1 = [
    "poi",
    "salary",
    "deferral_payments",
    "total_payments",
    "loan_advances",
    "bonus",
    "restricted_stock_deferred",
    "deferred_income",
    "total_stock_value",
    "expenses",
    "exercised_stock_options",
    "other",
    "long_term_incentive",
    "restricted_stock",
    "director_fees",
    "to_messages",
    "from_poi_to_this_person",
    "from_messages",
    "from_this_person_to_poi",
    "shared_receipt_with_poi"
]

# adaboost

# adaBoostClass = AdaBoostClassifier()
import tester
# tester.test_classifier(adaBoostClass, data_dict, feature1)


# ## Naive Bayes


# naiveBayesClass = GaussianNB()
# tester.test_classifier(naiveBayesClass, data_dict, feature1)


# ## SVC



SVCClass = SVC(kernel='linear', max_iter=1000)
# tester.test_classifier(SVCClass, data_dict, feature1)


feature2 = [
    "poi",
    "salary",
    "total_payments",
    "bonus",
    "deferred_income",
    "total_stock_value",
    "expenses",
    "exercised_stock_options",
    "other",
    "long_term_incentive",
    "restricted_stock",
    "to_messages",
    "from_poi_to_this_person",
    "from_messages",
    "from_this_person_to_poi",
    "shared_receipt_with_poi"
]


# tester.test_classifier(adaBoostClass, data_dict, feature2)

# tester.test_classifier(naiveBayesClass, data_dict, feature2)

# tester.test_classifier(SVCClass, data_dict, feature2)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


def feat_eng(data_dict, feat1, feat2, newfeat):
    """
    Creates a new feature by using 2 existing features
    Args:
    data_dict: dictionary of data
    feat1: feature 1
    feat2: feature 2
    newfeat: new feature
    Output: dictionary of the new feature
    """
    for person in data_dict:
        if data_dict[person][feat2] == 0:
            data_dict[person][newfeat] = 0.0
        elif data_dict[person][feat1] == "NaN" or data_dict[person][feat2] == "NaN":
            data_dict[person][newfeat] = "NaN"
        else:
            data_dict[person][newfeat] = float(
                data_dict[person][feat1]) / float(data_dict[person][feat2])

    return data_dict

data_dict = feat_eng(data_dict,
                     "bonus",
                     "total_payments",
                     "bonus_rate")

data_dict = feat_eng(data_dict,
                     "salary",
                     "total_payments",
                     "salary_rate")

data_dict = feat_eng(data_dict,
                     "total_stock_value",
                     "total_payments",
                     "stock_rate")

data_dict = feat_eng(data_dict,
                     "from_this_person_to_poi",
                     "from_messages",
                     "from_rate")

data_dict = feat_eng(data_dict,
                     "from_poi_to_this_person",
                     "to_messages",
                     "to_rate")


feature3 = [
    "poi",
    "salary",
    "deferral_payments",
    "total_payments",
    "loan_advances",
    "bonus",
    "restricted_stock_deferred",
    "deferred_income",
    "total_stock_value",
    "expenses",
    "exercised_stock_options",
    "other",
    "long_term_incentive",
    "restricted_stock",
    "director_fees",
    "to_messages",
    "from_poi_to_this_person",
    "from_messages",
    "from_this_person_to_poi",
    "shared_receipt_with_poi",
    "bonus_rate",
    "salary_rate",
    "stock_rate",
    "from_rate",
    "to_rate"
]


# tester.test_classifier(adaBoostClass, data_dict, feature3)

# tester.test_classifier(naiveBayesClass, data_dict, feature3)


# tester.test_classifier(SVCClass, data_dict, feature3)


def kbest(data_dict, features_list):
    """
    Selects the best features
    Args:
    data_dict: dictionary of data
    features: list of features
    Output: dictionary of the new feature
    """
    data = featureFormat(data_dict, features_list, sort_keys=True)
    # Split labels (poi) from other features
    targets, features = targetFeatureSplit(data)

    # Set up the scaler
    minmax_scaler = preprocessing.MinMaxScaler()
    features_minmax = minmax_scaler.fit_transform(features)

    # k is selected 10 --
    k_best = SelectKBest(chi2, k=10)

    # Use the instance to extract the k best features
    features_kbest = k_best.fit_transform(features_minmax, targets)

    scores = ['%.2f' % elem for elem in k_best.scores_]

    # Round the values
    feature_scores_pvalues = ['%.3f' % elem for elem in k_best.pvalues_]

    # Create an array of feature names, scores and pvalues
    k_features = [
        (features_list[i+1],
                   scores[i],
                   feature_scores_pvalues[i])
            for i in k_best.get_support(indices=True)]

    # Sort the array by score
    k_features = sorted(k_features, key=lambda f: float(f[1]))
    print(k_features)

    return None


# kbest(data_dict, feature1)


featuresKBest = [
    "poi",
    "director_fees",
    "other",
    "shared_receipt_with_poi",
    "long_term_incentive",
    "total_payments",
    "salary",
    "bonus",
    "total_stock_value",
    "loan_advances",
    "exercised_stock_options"
]


# tester.test_classifier(adaBoostClass, data_dict, featuresKBest)


# tester.test_classifier(naiveBayesClass, data_dict, featuresKBest)




# tester.test_classifier(SVCClass, data_dict, featuresKBest)

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


def feature_scalars(d, features_list, test_size, random_state=42):
    """
    Gives every feature a scalar
    """
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    target, features = targetFeatureSplit(data)

    # Create both training and test sets through split_data()
    features_train, features_test, labels_train, labels_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state)

    classifier = ["ADA", "SVC"]
    for c in classifier:
        if c == "ADA":
            clf = AdaBoostClassifier()
        elif c == "SVM":
            clf = SVC(kernel='linear', max_iter=1000)

        result = []
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        importances = clf.feature_importances_

        for i in range(len(importances)):
            t = [features_list[i], importances[i]]
            result.append(t)

        result = sorted(result, key=lambda x: x[1], reverse=True)

        print(result)

    return None


#feature_scalars(data_dict, feature3, 0.35)


feature4SVC = [
    "poi",
    "salary",
    "deferral_payments",
    "loan_advances",
    "restricted_stock_deferred",
    "total_stock_value",
    "exercised_stock_options",
    "other",
    "long_term_incentive",
    "director_fees",
    "to_messages",
    "from_poi_to_this_person",
    "from_messages",
    "from_this_person_to_poi",
    "shared_receipt_with_poi",
    "bonus_rate",
    "salary_rate",
    "stock_rate",
    "from_rate",
    "to_rate"
]


# clf_SVC = SVC(kernel='linear', max_iter=1000)
# tester.test_classifier(clf_SVC, data_dict, feature4SVC)


def tune_SVC(d, features_list, scaler=True):
    """
    Prints the results of tuning process.
    """
    # Strip the values
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])

    parameters = ([{'svm__C': [1, 50, 100, 1000],
                    'svm__gamma': [0.5, 0.1, 0.01],
                    'svm__degree': [1, 2],
                    'svm__kernel': ['rbf', 'poly', 'linear'],
                    'svm__max_iter': [1, 100, 1000]}])

    svm_clf = GridSearchCV(svm,
                           parameters,
                           scoring='f1').fit(
        features, labels).best_estimator_

    return None


# tune_SVC(data_dict, feature4SVC)


def get_svc(d, features_list):
    """
    Generates the classifier for final submission.
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])

    parameters = ([{'svm__C': [50],
                    'svm__gamma': [0.1],
                    'svm__degree': [2],
                    'svm__kernel': ['poly'],
                    'svm__max_iter': [100]}])

    svm_clf = GridSearchCV(svm,
                           parameters,
                           scoring='f1').fit(
        features, labels).best_estimator_

    return svm_clf




def test_clf(d, features_list, random_state=42):
    """
    Returns test results
    """
    # Keep only the values from features_list
    data = featureFormat(d, features_list, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)

    test_sizes = [0.2, 0.4, 0.6]

    for test_size in test_sizes:
        # Create both training and test sets through split_data()
        features_train, features_test, labels_train, labels_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state)

        clf = get_svc(d, features_list)

        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)

        print("# METRICS FOR TEST SIZE OF:", test_size)
        acc = accuracy_score(labels_test, pred)
        print("* Accuracy:", acc)

        pre = precision_score(labels_test, pred)
        print("* Precision:", pre)

        rec = recall_score(labels_test, pred)
        print("* Recall:", rec)
        print("\n")

    return

# Example starting point. Try investigating other evaluation techniques!

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

my_classifier = get_svc(data_dict, feature4SVC)
my_dataset = data_dict
my_feature_list = feature4SVC

dump_classifier_and_data(my_classifier, my_dataset, my_feature_list)

test_clf(data_dict, feature4SVC, random_state=42)

