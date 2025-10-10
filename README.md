# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
3. 
## Algorithm
1. Load and preprocess the dataset.
2. Split the data into training and testing sets.
3. Convert text messages into numerical features.
4. Train the SVM model and make predictions.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: shaalini.s
RegisterNumber:  25017649
*/
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

 
# 'Label': 1 = Spam, 0 = Not Spam
data = {
    'Email': [
        "Congratulations! You won a lottery. Claim now.",
        "Meeting tomorrow at 10am.",
        "Free tickets for the concert. Click here!",
        "Please review the attached report.",
        "You have been selected for a free gift.",
        "Lunch at 1pm?",
        "Earn money fast with this simple trick.",
        "Project deadline extended to next week."
    ],
    'Label': [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
print("Dataset:\n", df, "\n")


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Email'])  # Bag-of-words representation
y = df['Label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC(kernel='linear')  # Linear kernel is common for text classification
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


new_emails = [
    "Win a free iPhone now!",
    "Can we reschedule the meeting to 3pm?"
]
new_X = vectorizer.transform(new_emails)
predictions = model.predict(new_X)

for email, label in zip(new_emails, predictions):
    print(f"\nEmail: '{email}'")
    if label == 1:
        print(" This is SPAM.")
    else:
        print(" This is NOT SPAM.")
```

## Output:
![SVM For Spam Mail Detection](sam.png)
![alt text](<Screenshot 2025-10-07 195128.png>)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
