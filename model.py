import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import  matplotlib.pyplot as plt

# ---------------------------
# Load Your Dataset
# ---------------------------
df = pd.read_csv("your_dataset.csv")  # Change filename if needed

# Convert label to 0 (ham) and 1 (spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ---------------------------
# Preprocessing
# ---------------------------
stopwords = set([
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "from", "of", "is", "are",
    "was", "were", "be", "been", "has", "have", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "this", "that", "these", "those", "with", "as", "by", "not", "but"
])

def preprocess(text):
    text = str(text).lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

df['cleaned'] = df['message'].apply(preprocess)

# ---------------------------
# TF-IDF Vectorization
# ---------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# ---------------------------
# Split and Train
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------------------
# Evaluate
# ---------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot True vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='True', linestyle='-', color='blue')
plt.plot(y_pred, label='Predicted', linestyle='--', color='red')
plt.title('True vs Predicted')
plt.xlabel('Samples')
plt.ylabel('Label')
plt.legend()
plt.show()

# ---------------------------
# Save Model & Vectorizer
# ---------------------------
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
