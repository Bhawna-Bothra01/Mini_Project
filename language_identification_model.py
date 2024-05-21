import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib


nltk.download('punkt')
nltk.download('stopwords')

# Manual NLTK data path configuration
nltk.data.path.append(r"C:\nltk_data")

# Load JSON data
with open(r'C:\Users\hp\OneDrive\Desktop\mini_project\bhasha-abhijnaanam.json', 'r') as f:
    json_data = json.load(f)
    data = json_data["data"]  # Access the data under the "data" key

# Initialize Snowball Stemmer for English
stemmer = SnowballStemmer("english")

# Preprocessing function
def preprocess(text):
    # Remove non-text characters
    text = re.sub(r"[^\w\s]", "", text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

# # Preprocess each native sentence
# for item in data:
#     preprocessed_tokens = preprocess(item["native sentence"])
#     print("Native Sentence:", item["native sentence"])
#     print("Preprocessed Tokens:", preprocessed_tokens)
#     print()

# Initialize lists to store preprocessed tokens and language labels
X = []
y = []

# Preprocess each document and assign language labels
for item in data:
    preprocessed_tokens = preprocess(item["native sentence"])
    X.append(preprocessed_tokens)
    y.append(item["language"])

# Flatten the list of preprocessed tokens into a list of strings
X_flattened = [' '.join(tokens) for tokens in X]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.25, random_state=42)

# Convert tokens into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Support Vector Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = svm_classifier.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

classifier = open("svm_classifier.pkl","wb")
vectorizer = open("tfidf_vectorizer.pkl","wb")

joblib.dump(svm_classifier,'svm_classifier.pkl')
joblib.dump(tfidf_vectorizer,'tfidf_vectorizer.pkl')

classifier.close()
vectorizer.close()