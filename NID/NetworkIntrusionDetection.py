import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=3)
sns.set(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

train = pd.read_csv("../input/Train_data.csv")
test = pd.read_csv("../input/Test_data.csv")

train.describe()

print(train['num_outbound_cmds'].value_counts())
print(test['num_outbound_cmds'].value_counts())

train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)

train['class'].value_counts()

import sklearn

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

cols = train.select_dtypes(include=['float64','int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
sc_test = scaler.fit_transform(test.select_dtypes(include=['float64','int64']))

sc_traindf = pd.DataFrame(sc_train, columns = cols)
sc_testdf = pd.DataFrame(sc_test, columns = cols)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

cattrain = train.select_dtypes(include=['object']).copy()
cattest = test.select_dtypes(include=['object']).copy()

traincat = cattrain.apply(encoder.fit_transform)
testcat = cattest.apply(encoder.fit_transform)

enctrain = traincat.drop(['class'], axis=1)
cat_Ytrain = traincat[['class']].copy()

train_x = pd.concat([sc_traindf,enctrain],axis=1)
train_y = train['class']
test_df = pd.concat([sc_testdf,testcat],axis=1)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier();

rfc.fit(train_x, train_y);
score = np.round(rfc.feature_importances_,3)
importances = pd.DataFrame({'feature':train_x.columns,'importance':score})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
plt.rcParams['figure.figsize'] = (11, 4)
importances.plot.bar();

from sklearn.feature_selection import RFE
import itertools
rfc = RandomForestClassifier()

rfe = RFE(rfc, n_features_to_select=15)
rfe = rfe.fit(train_x, train_y)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_x.columns)]
selected_features = [v for i, v in feature_map if i==True]

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(train_x,train_y,train_size=0.70, random_state=2)

from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
KNN_Classifier.fit(X_train, Y_train);

LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
LGR_Classifier.fit(X_train, Y_train);

BNB_Classifier = BernoulliNB()
BNB_Classifier.fit(X_train, Y_train)

DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
DTC_Classifier.fit(X_train, Y_train)

models = []
models.append(('Naive Baye Classifier', BNB_Classifier))
models.append(('Decision Tree Classifier', DTC_Classifier))
models.append(('KNeighborsClassifier', KNN_Classifier))
models.append(('LogisticRegression', LGR_Classifier))

for i, v in models:
    scores = cross_val_score(v, X_train, Y_train, cv=10)
    accuracy = metrics.accuracy_score(Y_train, v.predict(X_train))
    confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_train))
    classification = metrics.classification_report(Y_train, v.predict(X_train))
    print()
    print('============================== {} Model Evaluation =============================='.format(i))
    print()
    print ("Cross Validation Mean Score:" "\n", scores.mean())
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification)
    print()

for i, v in models:
    accuracy = metrics.accuracy_score(Y_test, v.predict(X_test))
    confusion_matrix = metrics.confusion_matrix(Y_test, v.predict(X_test))
    classification = metrics.classification_report(Y_test, v.predict(X_test))
    print()
    print('============================== {} Model Test Results =============================='.format(i))
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification)
    print()

pred_knn = KNN_Classifier.predict(test_df)
pred_NB = BNB_Classifier.predict(test_df)
pred_log = LGR_Classifier.predict(test_df)
pred_dt = DTC_Classifier.predict(test_df)

def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_pred)):
        if y_actual.iat[i] == y_pred[i] == 'anomaly':
            TP += 1
        if y_pred[i] == 'anomaly' and y_actual.iat[i] != y_pred[i]:
            FP += 1
        if y_actual.iat[i] == y_pred[i] == 'normal':
            TN += 1
        if y_pred[i] == 'normal' and y_actual.iat[i] != y_pred[i]:
            FN += 1
    return (TP, FP, TN, FN)

for i, v in models:
    print("For model:", i)
    TP, FP, TN, FN = perf_measure(Y_test, v.predict(X_test))
    print("TP:", TP, "\tFP:", FP, "\t\tTN:", TN, "\tFN:", FN)
    print()

for i, v in models:
    print("For model: ", i)
    print ("Expected: ", Y_test.iloc[2], "\tPredicted: ", v.predict(X_test).reshape(1, -1)[0][2] )
    print()


def find_FP(y_actual, y_pred):
    FP = []
    for i in range(len(y_pred)):
        if y_pred[i] == 'anomaly' and y_actual.iat[i] != y_pred[i]:
            FP.append(i)
    return (pd.Series(FP))

def find_FN(y_actual, y_pred):
    FN = []
    for i in range(len(y_pred)):
        if y_pred[i] == 'normal' and y_actual.iat[i] != y_pred[i]:
            FN.append(i)
    return (pd.Series(FN))

FP_NB= find_FP(Y_test, models[0][1].predict(X_test))
print("Size of number of FP:", FP_NB.size)
FN_NB= find_FN(Y_test, models[0][1].predict(X_test))
print("Size of number of FN:", FN_NB.size)

X_test_subset = []
Y_test_subset = []
for i in FP_NB:
    X_test_subset.append(X_test.iloc[i])
    Y_test_subset.append(Y_test.iat[i])
for i in FN_NB:
    X_test_subset.append(X_test.iloc[i])
    Y_test_subset.append(Y_test.iat[i])

X_test_sub = pd.DataFrame(X_test_subset)
Y_test_sub = pd.Series(Y_test_subset)
print("Size of X_test_sub:", X_test_sub.shape[0])
print("Size of Y_test_sub:", Y_test_sub.size)

accuracy = metrics.accuracy_score(Y_test_sub, models[1][1].predict(X_test_sub))
confusion_matrix = metrics.confusion_matrix(Y_test_sub, models[1][1].predict(X_test_sub))
classification = metrics.classification_report(Y_test_sub, models[1][1].predict(X_test_sub))
print()
print('============================== {} Model Test Results =============================='.format("NB -> DT"))
print()
print ("Model Accuracy:" "\n", accuracy)
print()
print("Confusion matrix:" "\n", confusion_matrix)
print()
print("Classification report:" "\n", classification)
print()

print("For Naive Bayes:")
TP_old, FP_old, TN_old, FN_old = perf_measure(Y_test, BNB_Classifier.predict(X_test))
print ("TP:", TP_old, "\tFP:", FP_old, "\t\tTN:", TN_old, "\tFN:", FN_old)

print()
print("For Naive Bayes -> Decision Tress:")
TP_new, FP_new, TN_new, FN_new = perf_measure(Y_test_sub, DTC_Classifier.predict(X_test_sub))
print ("TP:", TP_new, "\tFP:", FP_new, "\t\tTN:", TN_new, "\tFN:", FN_new)

print()
print("For Naive Bayes + Decision Tress:")
tp = TP_old +TP_new
fp = FP_new
tn = TN_old +TN_new
fn = FN_new
print ("TP:", tp, "\tFP:", fp, "\t\tTN:", tn, "\tFN:", fn)

acc_old= (TP_old + TN_old) / (TP_old + FP_old + TN_old + FN_old)
mis_old= (FP_old + FN_old) / (TP_old + FP_old + TN_old + FN_old)
prec_old= TP_old / (TP_old + FP_old)
sen_old= TP_old / (TP_old + FN_old)
spec_old= TN_old / (TN_old + FP_old)

acc= (tp + tn) / (tp + fp + tn + fn)
mis= (fp + fn) / (tp + fp + tn + fn)
prec= tp / (tp + fp)
sen= tp / (tp + fn)
spec= tn / (tn + fp)

print ("Accuracy")
print ("Old: ", acc_old, "\tNew: ", acc)
print ("\nMisclassification")
print ("Old: ", mis_old, "\tNew: ", mis)
print ("\nPrecision")
print ("Old: ", prec_old, "\tNew: ", prec)
print ("\nSensitivity")
print ("Old: ", sen_old, "\tNew: ", sen)
print ("\nSpecificity")
print ("Old: ", spec_old, "\tNew: ", spec)

barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

NB = [TP_old, FP_old, TN_old, FN_old]
NBandDT = [tp, fp, tn, fn]

br1 = np.arange(len(NB))
br2 = [x + barWidth for x in br1]

plt.bar(br1, NB, color='b', width=barWidth, edgecolor='grey', label='Naive Bayes')
plt.bar(br2, NBandDT, color='g', width=barWidth, edgecolor='grey', label='Naive Bayes and Decision Tree')

plt.xlabel('Confusion Matrix Element', fontweight='bold', fontsize=15)
plt.ylabel('Value', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(NB))], ['TP', 'FP', 'TN', 'FN'])

plt.legend()
plt.title("Confusion Matrix")
plt.show()

barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

Old = [acc_old, mis_old, prec_old, sen_old, spec_old]
New = [acc, mis, prec, sen, spec]

br1 = np.arange(len(Old))
br2 = [x + barWidth for x in br1]

plt.bar(br1, Old, color='b', width=barWidth, edgecolor='grey', label='Old')
plt.bar(br2, New, color='g', width=barWidth, edgecolor='grey', label='New')

plt.xlabel('Performance Metrics', fontweight='bold', fontsize=15)
plt.ylabel('Value', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(NB))],
           ['Accuracy', 'Misclassification', 'Precision', 'Sensitivity', 'Specificity'])

plt.legend()
plt.title("Comparison of performance metrics")
plt.show()