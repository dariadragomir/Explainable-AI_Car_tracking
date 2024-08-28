import codecs
import zipfile
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, cohen_kappa_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

option = input("Multinomial Naive Bayes = 1; Mașini cu vectori suport = 2; Regressie Logistică = 3; Nearest Neighbors = 4; Arbori de decizie = 5; Stochastic Gradient Descent = 6; Kernel Ridge Regression = 7; Probability Calibration = 8. Introduceti optiunea:")

def writeResults(testIDX, labels):
    with open("kaggle.csv", "w") as output:
        output.write("id,label\n")
        for p in zip(testIDX, labels):
            output.write(str(p[0])+','+str(p[1])+'\n')

def readSamples(zipFile, textFile):
    data = []
    with zipfile.ZipFile(zipFile) as thezip:
        with thezip.open(textFile, mode='r') as thefile:
            for line in codecs.iterdecode(thefile, 'utf8'):
                idAndSentence = line.split("\t")
                id, sentence = int(idAndSentence[0].strip()), str(idAndSentence[1].strip())
                dataLine = {'id' : id, 'sentence': sentence }
                data.append(dataLine)
    return pd.DataFrame(data)

def readLabels(filePath):
    data = []
    with open(filePath, mode='r',encoding="UTF-8") as thefile:
        for line in thefile.readlines():
            idAndLabel = line.split("\t")
            id, label = int(idAndLabel[0].strip()), int(idAndLabel[1].strip())
            dataLine = {'id': id, 'language': label}
            data.append(dataLine)
    return pd.DataFrame(data)

train_samples = readSamples(zipFile="train_samples.txt.zip", textFile="train_samples.txt")
train_labels = readLabels(filePath="train_labels.txt")
validation_samples = readSamples(zipFile="validation_samples.txt.zip",textFile="validation_samples.txt")
validation_labels = readLabels(filePath="validation_labels.txt")
test_samples = readSamples(zipFile= "test_samples.txt.zip", textFile="test_samples.txt")

vectorizer = CountVectorizer(analyzer='char', ngram_range=(2,8), binary=True, encoding="UTF-8", lowercase=False, strip_accents="unicode")

train_samples = train_samples.drop('id', axis=1)
train_labels = train_labels.drop('id', axis=1)
validation_labels = validation_labels.drop('id', axis=1)
validation_samples = validation_samples.drop('id', axis=1)
testIDX = test_samples.drop('sentence', axis=1)

train_samples = vectorizer.fit_transform(train_samples['sentence'])
validation_samples = vectorizer.transform(validation_samples['sentence'])
test_samples = vectorizer.transform(test_samples['sentence'])


if option == '1':
    classifier = MultinomialNB(alpha=0.2, fit_prior=True)
elif option == '2':
    classifier = svm.SVC(C=5, kernel="linear", gamma="scale")
elif option == '3':
    classifier = LogisticRegression(max_iter=1000)
elif option == '4':
    classifier = KNeighborsClassifier(n_neighbors=5)
elif option == '5':
    classifier = DecisionTreeClassifier(random_state=0)
elif option == '6':
    classifier = SGDClassifier(max_iter=1000, tol=1e-3, random_state=0)
elif option == '7':
    classifier = KernelRidge(alpha=1.0, kernel='rbf')
elif option == '8':
    base_classifier = LogisticRegression(max_iter=1000)
    classifier = CalibratedClassifierCV(base_classifier, method='sigmoid')


f1 = []

for i in range(0, 3):
    classifier.fit(train_samples, train_labels.values.ravel())

    if option == '8':
        predictions = classifier.predict(validation_samples)
    else:
        predictions = classifier.predict(validation_samples)
        predictions = np.rint(predictions).astype(int)  

    f1.append(f1_score(validation_labels, predictions, average='macro'))
    accuracy = accuracy_score(validation_labels, predictions)
    precision = precision_score(validation_labels, predictions, average='macro')
    recall = recall_score(validation_labels, predictions, average='macro')
    kappa = cohen_kappa_score(validation_labels, predictions)
    balanced_acc = balanced_accuracy_score(validation_labels, predictions)


std_dev_f1 = np.std(f1)

print(f"F1 Score: {f1}")
print(f"Accuracy Score: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Cohen Kappa: {kappa}")
print(f"Balanced Accuracy: {balanced_acc}")
print(f"Standard Deviation of Predictions: {std_dev_f1}")

# predictii pt setul de test
labels = classifier.predict(test_samples)
if option != '8':
    labels = np.rint(labels).astype(int)  
writeResults(testIDX['id'], labels)

print(confusion_matrix(validation_labels, predictions))
