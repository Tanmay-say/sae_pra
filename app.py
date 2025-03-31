import re
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load model, tokenizer, and scaler
try:
    model = load_model('best_model.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('length_features_scaler.pickle', 'rb') as f:
        length_features_scaler = pickle.load(f)
except Exception as e:
    print(f"Error loading model or files: {e}")
    exit(1)

MAX_QUESTION_LENGTH = 100
MAX_DESIRED_LENGTH = 100
MAX_STUDENT_LENGTH = 100

def preprocess_text(text):
    """Cleans and preprocesses the input text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/demo', methods=['GET', 'POST'])
def demo():
    result = None
    if request.method == 'POST':
        question = request.form.get('question', '')
        desired_answer = request.form.get('desired_answer', '')
        student_answer = request.form.get('student_answer', '')

        if not question or not desired_answer or not student_answer:
            result = {
                'score': 1.0,
                'feedback': "Please provide all inputs."
            }
        else:
            score = evaluate_answer(question, desired_answer, student_answer)
            result = {
                'score': score,
                'feedback': generate_feedback(score)
            }

    return render_template('demo.html', result=result)

def evaluate_answer(question, desired_answer, student_answer):
    """Evaluates the student answer based on model predictions."""
    try:
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

        length_features = np.array([[question_length, desired_length, student_length, length_ratio]])
        length_features_scaled = length_features_scaler.transform(length_features)

        # Predict score
        prediction = model.predict([question_padded, desired_padded, student_padded, length_features_scaled])
        score = np.clip(prediction[0][0], 1.0, 5.0)  # Ensure score is between 1-5
        return round(score, 1)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1.0  # Return minimum score in case of an error

def generate_feedback(score):
    """Generates feedback based on the score."""
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
