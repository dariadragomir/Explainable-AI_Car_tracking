import os
import zipfile
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from google.colab import files

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def load_images_and_labels(folder_path, label_map):
    data = []
    labels = []
    for label_name, label in label_map.items():
        path = os.path.join(folder_path, label_name)
        for file in os.listdir(path):
            if file.endswith(".jpg"):
                file_path = os.path.join(path, file)
                image = Image.open(file_path).convert('RGB')
                image = image.resize((128, 128))  # Resize for uniformity
                data.append(np.array(image).flatten())
                labels.append(label)
    return np.array(data), np.array(labels)


print("Please upload the 'train.zip' file.")
train_zip = files.upload()

print("Please upload the 'test1.zip' file.")
test1_zip = files.upload()

extract_zip('train.zip', 'train')
extract_zip('test1.zip', 'test1')

label_map = {'cat': 0, 'dog': 1}
X, y = load_images_and_labels('train', label_map)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Multinomial Naive Bayes': make_pipeline(StandardScaler(with_mean=False), MultinomialNB()),
    'Support Vector Machine': make_pipeline(StandardScaler(), SVC(probability=True)),
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    'Decision Tree': DecisionTreeClassifier(),
    'Stochastic Gradient Descent': make_pipeline(StandardScaler(), SGDClassifier())
}


results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    std_dev = np.std(cross_val_score(model, X, y, cv=5))
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Balanced Accuracy': balanced_accuracy,
        'Standard Deviation': std_dev
    }

for model_name, metrics in results.items():
    print(f"\nModel: {model_name}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
