from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import string

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Define the same preprocessing function
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        cleaned_message = preprocess(message)
        vectorized_message = vectorizer.transform([cleaned_message])
        prediction = model.predict(vectorized_message)
        
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        return render_template('result.html', message=message, result=result)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    message = data['message']
    cleaned_message = preprocess(message)
    vectorized_message = vectorizer.transform([cleaned_message])
    prediction = model.predict(vectorized_message)
    
    return jsonify({'prediction': int(prediction[0]), 'message': message})

if __name__ == '__main__':
    app.run(debug=True)