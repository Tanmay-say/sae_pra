import re
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load model, tokenizer, and scaler
model = load_model('best_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('length_features_scaler.pickle', 'rb') as f:
    length_features_scaler = pickle.load(f)

MAX_QUESTION_LENGTH = 100  # Update as per your model
MAX_DESIRED_LENGTH = 100
MAX_STUDENT_LENGTH = 100

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo', methods=['GET', 'POST'])
def demo():
    result = None
    if request.method == 'POST':
        question = request.form.get('question')
        desired_answer = request.form.get('desired_answer')
        student_answer = request.form.get('student_answer')

        score = evaluate_answer(question, desired_answer, student_answer)

        result = {
            'score': score,
            'feedback': generate_feedback(score)
        }

    return render_template('demo.html', result=result)

def evaluate_answer(question, desired_answer, student_answer):
    if not question or not desired_answer or not student_answer:
        return 1.0  # Minimum grade

    # Preprocess text
    question_clean = preprocess_text(question)
    desired_clean = preprocess_text(desired_answer)
    student_clean = preprocess_text(student_answer)

    # Tokenize and pad sequences
    question_seq = tokenizer.texts_to_sequences([question_clean])
    desired_seq = tokenizer.texts_to_sequences([desired_clean])
    student_seq = tokenizer.texts_to_sequences([student_clean])

    question_padded = pad_sequences(question_seq, maxlen=MAX_QUESTION_LENGTH, padding='post')
    desired_padded = pad_sequences(desired_seq, maxlen=MAX_DESIRED_LENGTH, padding='post')
    student_padded = pad_sequences(student_seq, maxlen=MAX_STUDENT_LENGTH, padding='post')

    # Compute length features
    question_length = len(question_clean.split())
    desired_length = len(desired_clean.split())
    student_length = len(student_clean.split())
    length_ratio = student_length / (desired_length + 1)
    length_features = scaler.transform([[question_length, desired_length, student_length, length_ratio]])

    # Predict score
    prediction = model.predict([question_padded, desired_padded, student_padded, length_features])
    score = np.clip(prediction[0][0], 1.0, 5.0)  # Ensure score is between 1-5
    return round(score, 1)

def generate_feedback(score):
    if score >= 4.5:
        return "Excellent! Demonstrates comprehensive understanding."
    elif score >= 4.0:
        return "Very good! Covers most key concepts."
    elif score >= 3.0:
        return "Good. Addresses main points but needs more depth."
    elif score >= 2.0:
        return "Fair. Partial understanding demonstrated."
    else:
        return "Needs improvement. Significant gaps in content."

if __name__ == '__main__':
    app.run(debug=True)
