from flask import Flask, request, jsonify, render_template
import joblib
import librosa
import numpy as np
import os
import io
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import langid
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load the trained models and vectorizer
voice_model = joblib.load('trained_language_model.pkl')
joblib_in1 = open("svm_classifier.pkl", "rb")
joblib_in2 = open("tfidf_vectorizer.pkl", "rb")
svm_classifier = joblib.load(joblib_in1)
tfidf_vectorizer = joblib.load(joblib_in2)

# Load the DataFrame from the .pkl file
with open("language_predictions.pkl", "rb") as f:
    df = pickle.load(f)

# Mapping from numeric labels to language names for voice model
lang_label_dict = {
    0: 'Punjabi', 1: 'Tamil', 2: 'Hindi', 3: 'Bengali',
    4: 'Telugu', 5: 'Kannada', 6: 'Gujarati', 7: 'Urdu',
    8: 'Marathi', 9: 'Malayalam'
}

# Language codes to names for code-switching model
language_names = {
    'kn': 'Kannada',
    'en': 'English',
    'hi': 'Hindi',
    'ne': 'Nepali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'bn': 'Bangla',
    'ml': 'Malayalam',
    'as': 'Assamese',
    'pa': 'Punjabi',
    'ur': 'Urdu',
    'gu': 'Gujarati',
    'sa': 'Sanskrit',
    'kok': 'Konkani',
    'mni': 'Manipuri',
    'brx': 'Bodo',
    'mai': 'Maithili',
    'doi': 'Dogri',
    'ks': 'Kashmiri',
    'or': 'Oriya'
}

def extract_features(audio_data, sample_rate):
    audio = librosa.util.normalize(audio_data)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs, axis=1)
    features = mfccs_mean.reshape(1, -1)
    return features

@app.route("/predict_voice/", methods=["POST"])
def predict_voice_language():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.content_type not in ["audio/mpeg", "audio/wav"]:
            return jsonify({"error": "Invalid file type. Only MP3 and WAV are supported."}), 400

        audio_data, sample_rate = librosa.load(io.BytesIO(file.read()), sr=None)
        features = extract_features(audio_data, sample_rate)
        prediction = voice_model.predict(features)
        predicted_language = lang_label_dict.get(prediction[0], "Unknown Language")
        return jsonify({"predicted_language": predicted_language})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Preprocessing function
def preprocess(text):
    text = re.sub(r"[^\w\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
    tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    tokens = [SnowballStemmer("english").stem(word) for word in tokens]
    tokens = [word.lower() for word in tokens]
    return ' '.join(tokens)

# Language prediction function for code-switching
def predict_code_switch_language(sentence):
    langs = set()
    for word in sentence.split():
        lang = langid.classify(word)[0]
        langs.add(lang)
    return list(langs)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route("/predict_code_switch/", methods=["POST"])
def get_code_switch_language_predictions():
    try:
        data = request.json
        sentence = data['sentence']
        preprocessed = preprocess(sentence)
        languages = predict_code_switch_language(sentence)
        language_names_list = [language_names[lang] if lang in language_names else lang for lang in languages]
        return jsonify({"languages": language_names_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_text_language/", methods=["POST"])
def predict_text_language():
    try:
        data = request.json
        preprocessed_text = preprocess(data['text'])
        text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
        prediction = svm_classifier.predict(text_tfidf)
        return jsonify({"predicted_language": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/data/", methods=["GET"])
def get_data():
    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
    from waitress import serve
    serve(app, host='0.0.0.0', port=8000)
