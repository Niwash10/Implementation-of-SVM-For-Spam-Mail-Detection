# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset.
2. Convert text into numerical features using TF-IDF.
3. Train the SVM classifier.
4. Test and evaluate the model accuracy.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: NIWASH K
RegisterNumber:  212225230205
*/
```

```
# Exp 11 – SVM for Spam Mail Detection with Visualization

# 1. Import Required Libraries
import chardet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Detect File Encoding
file_path = r"C:\Users\acer\Downloads\spam.csv"   # use "/content/spam.csv" for Colab

with open(file_path, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

print("Detected Encoding:", result)

# 3. Load Dataset
data = pd.read_csv(file_path, encoding=result['encoding'])

# 4. Basic Data Exploration
print(data.head())
print(data.info())
print(data.isnull().sum())

# 5. Visualization: Spam vs Ham Distribution
plt.figure(figsize=(5,4))
sns.countplot(x=data['v1'])
plt.title("Distribution of Spam and Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.show()

# 6. Message Length Visualization
data['msg_length'] = data['v2'].apply(len)

plt.figure(figsize=(6,4))
sns.histplot(data=data, x='msg_length', hue='v1', bins=50, kde=True)
plt.title("Message Length Distribution (Spam vs Ham)")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

# 7. Feature and Target Selection
x = data['v2'].values     # messages
y = data['v1'].values     # labels

# 8. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# 9. Text Vectorization (Bag of Words)
cv = CountVectorizer()
x_train_vec = cv.fit_transform(x_train)
x_test_vec = cv.transform(x_test)

# 10. Initialize and Train SVM
svc = SVC(kernel='linear')
svc.fit(x_train_vec, y_train)

# 11. Prediction
y_pred = svc.predict(x_test_vec)

# 12. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 13. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix – SVM Spam Detection")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

```
## Output:
![SVM For Spam Mail Detection](sam.png)
<img width="996" height="816" alt="image" src="https://github.com/user-attachments/assets/a5efaf81-51c8-4cf1-affd-6bbb3d107e19" />

<img width="472" height="391" alt="download" src="https://github.com/user-attachments/assets/34b31b54-a952-4095-bf78-667f5e5ebf9e" />
<img width="549" height="391" alt="download" src="https://github.com/user-attachments/assets/e9202cdc-2576-495b-9a25-03926fef4025" />
<img width="563" height="275" alt="image" src="https://github.com/user-attachments/assets/c9af3dcb-4bd0-4c89-950d-d3eb03a7f8b2" />

<img width="444" height="391" alt="download" src="https://github.com/user-attachments/assets/6c05d3c0-b825-4b06-80e5-9b203a0b0b1d" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
