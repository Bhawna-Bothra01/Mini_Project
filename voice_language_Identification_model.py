import os
import glob
import joblib
import librosa
import pandas as pd
# from glob import glob
import random
import numpy as np
from scipy.signal import welch
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

mainDirectory=r'C:\Users\hp\OneDrive\Desktop\mini_project\voice_dataset'

languages = []

for file in os.listdir(mainDirectory) :
    counter = 0;
    for subFiles in os.listdir(f'{mainDirectory}/{file}') :
        counter += 1;
    languages.append(file)
    #print(f'- {file} - {counter}')

#print(languages)

# Correctly formatted path declarations
bengali = r'C:\Users\hp\OneDrive\Desktop\mini_project\voice_dataset\Bengali'
malayalam = r'C:\Users\hp\OneDrive\Desktop\mini_project\voice_dataset\Malayalam'
tamil = r'C:\Users\hp\OneDrive\Desktop\mini_project\voice_dataset\Tamil'
telugu = r'C:\Users\hp\OneDrive\Desktop\mini_project\voice_dataset\Telugu'
gujarati = r'C:\Users\hp\OneDrive\Desktop\mini_project\voice_dataset\Gujarati'  # Corrected spelling of "Gujarati"
hindi = r'C:\Users\hp\OneDrive\Desktop\mini_project\voice_dataset\Hindi'
kannada = r'C:\Users\hp\OneDrive\Desktop\mini_project\voice_dataset\Kannada'
marathi = r'C:\Users\hp\OneDrive\Desktop\mini_project\voice_dataset\Marathi'
urdu = r'C:\Users\hp\OneDrive\Desktop\mini_project\voice_dataset\Urdu'
punjabi = r'C:\Users\hp\OneDrive\Desktop\mini_project\voice_dataset\Punjabi'


label = []
lang = []

for language in languages :
    for file in os.listdir(f'{mainDirectory}/{language}')[:1000] :
        label.append(file[:-4])
        lang.append(language)

dframe = pd.DataFrame(data = {'label' : label,'language' : lang},columns = ['label','language'])

#print(dframe.head())

def snr(audio_data):
    signal = np.array(audio_data.get_array_of_samples())
    signal_power = np.sum(signal**2) / len(signal)
    noise_power = np.sum((signal - np.mean(signal)) ** 2) / len(signal)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def has_distortion(audio_segment,threshold = 0.05) :
    audioFile = audio_segment.get_array_of_samples()
    audio = np.array(audioFile)

    rms = np.sqrt(np.mean(audio ** 2))

    if rms > threshold :
        return True
    return False


def uneven_frequency(sound,threshold = 0.1,sampling_rate=44100) :
    audio = sound.get_array_of_samples()

    f, psdArr = welch(audio, fs=sampling_rate, nperseg=2048)

    # maximum psd value of the array
    maxPsd = np.max(psdArr)

    if maxPsd > threshold :
        return False
    return True



def audio_quality(audio_file, threshold):
    sound = AudioSegment.from_file(audio_file)
    snrRatio = snr(sound)

    cond = [snrRatio < threshold, has_distortion(sound), uneven_frequency(sound)]

    for i in cond:
        if i is False:
            return False

    return True

def verify_language(audio_file, expLang):
    detectLang = detectLang(audio_file)

    if detectLang != expLang :
        return False

    return True

def clean_data(files, expLang):
    cleanedFiles = []

    for file in files:
        if audio_quality(file) or verify_language(file, expLang):
            cleanedFiles.append(file)

    return cleanedFiles

def normalize_volume(audio):
    return librosa.util.normalize(audio)

def audio_segmentation(audio, segment_duration=5, sample_rate=22050):
    samplePerSegment = segment_duration * sample_rate
    segments = []
    for i in range(0,len(audio),samplePerSegment) :
        segments.append(audio[i:i+samplePerSegment])

    return segments

def Feature_extract(mainDirectory, languages, split=2000):
    data = []

    for language in languages:
        language_dir = os.path.join(mainDirectory, language)
        # Correct use of glob to find files
        audio_files = glob.glob(os.path.join(language_dir, '*.mp3'))

        print(f"Processing {len(audio_files)} files in {language_dir}")  # Debug: Check number of files

        for audio_file in audio_files[:split]:  # Limit to 'split' number of files
            try:
                # Using librosa to load audio files
                audio, sample_rate = librosa.load(audio_file, sr=None)  # Load with original sample rate
                audio = librosa.util.normalize(audio)
                mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                mfccs_mean = np.mean(mfccs, axis=1)
                data.append((mfccs_mean, language))
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

    df = pd.DataFrame(data, columns=['Features', 'Class'])
    return df

dframe = Feature_extract(mainDirectory, languages)
dframe.head()

langLabel = {
    'Punjabi': 0,
    'Tamil': 1,
    'Hindi': 2,
    'Bengali': 3,
    'Telugu': 4,
    'Kannada': 5,
    'Gujarati': 6,
    'Urdu': 7,
    'Marathi': 8,
    'Malayalam': 9
}

dframe['Class'] = dframe['Class'].apply(lambda x : langLabel[x])

features = dframe.Features
X= np.array(features.tolist())
y= dframe.Class

print("X shape:", X.shape if hasattr(X, 'shape') else "Not loaded")
print("y shape:", y.shape if hasattr(y, 'shape') else "Not loaded")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

# for units in [1,10,100]:
#     clf= MLPClassifier(hidden_layer_sizes=[units],random_state=1).fit(X_train,y_train)
#     print('for hidden layer= {}'.format(units))
#     print(classification_report(y_test,clf.predict(X_test)))

best_model = None
best_accuracy = 0
for units in [1, 10, 100]:
    clf = MLPClassifier(hidden_layer_sizes=[units], random_state=1)
    clf.fit(X_train, y_train)
    print('for hidden layer= {}'.format(units))
    report = classification_report(y_test, clf.predict(X_test))
    print(report)

    # Store the best model
    current_accuracy = accuracy_score(y_test, clf.predict(X_test))
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_model = clf

# Save the best model
if best_model:
    joblib.dump(best_model, 'trained_language_model.pkl')
    print("Best model saved with accuracy: {:.2f}".format(best_accuracy))

accuracy_score(y_test,clf.predict(X_test))

def extract_features(audio_file, augment=False):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(audio_file, sr=None)
        if augment:
            # Data augmentation by adding noise
            noise_amt = 0.005 * np.random.normal(0, 1, len(audio))
            audio = audio + noise_amt

            # Data augmentation by shifting the sound wave
            shift_range = int(random.random() * 1000)
            audio = np.roll(audio, shift_range)

        # Normalize audio
        audio = librosa.util.normalize(audio)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)

        return mfccs_mean.reshape(1, -1)  # Reshape for single sample prediction
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None
    
model = joblib.load('trained_language_model.pkl')

# Mapping from numeric labels to language names
lang_label_dict = {
    0: 'Punjabi', 1: 'Tamil', 2: 'Hindi', 3: 'Bengali',
    4: 'Telugu', 5: 'Kannada', 6: 'Gujarati', 7: 'Urdu',
    8: 'Marathi', 9: 'Malayalam'
}

def extract_features(audio_file):
    # Load the audio file
    audio, sample_rate = librosa.load(audio_file, sr=None)
    audio = librosa.util.normalize(audio)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Reshape features for the model input
    features = mfccs_mean.reshape(1, -1)
    return features

def predict_language(audio_file, model, label_dict):
    # Extract features
    features = extract_features(audio_file)

    # Use the loaded model to predict the language
    prediction = model.predict(features)

    # Convert numeric prediction back to language name
    predicted_language = label_dict.get(prediction[0], "Unknown Language")

    return predicted_language

# Load the trained model
model = joblib.load('trained_language_model.pkl')