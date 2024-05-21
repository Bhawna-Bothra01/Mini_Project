import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import langid
import pickle

# Create the DataFrame
df = pd.DataFrame({"Sentences": ["This is a pen और यह एक किताब है",
              "ಇದು ಒಂದು ಪುಸ್ತಕ है और यह एक पेन है।",
              "நான் ఈ వారం విద్యార్థిని చేయబోతున్నాను.Good Evening",
              "I am going to the market এবং আমি বই কিনব।",
              "ഹിന്ദി Good ഹിന്ദുക്കളുടെ ഭാഷ ആണ്.",
              "my name is Aditya ನನ್ನ ಹೆಸರು ಆದಿತ್ಯ",
              "এটি একটি বই এবং এটি ఒక పుస్తకం.",
              "I લાઇક to read ગુજરાતી books और यह एक किताब है ನನ್ನ",
              "ನನಗೆ ಈ ಸಂಜೆ ಚಿಂತೆಯಿಲ್ಲ",
              "अहं संस्कृते वाक्यं लिखामि",
              "गांव में सभी जोनोम सुबुं होकर मान सन्मान और आरो की भावना से लाना जोनोम लायो का आयोजन करते हैं।",
              "बिसोरो मोजां- गाज्रि के নিকट পানি অধিকতর  और सूखा खाना",
              "আমি আহোম ভাষাৰ গুৱাহাটি অত্যন্ত ভাল পায়",
              "আজি আমি পুঁজিৰ রান্না পঁচিব।",
              "মই খাবৰ প্ৰতি ভাল পাওঁছো।",
              "আমি খাবার খেতে ভালোবাসি।",
              "नांगान हानाय आमि मिजा दाम।",
              "मुझे खाना खाना पसंद है।",
              "હું ખાવાનું પ્રેમ કરું છું",
              "ನಾನು ಆಹಾರ ತಿನುವುದನ್ನು ಇಷ್ಟಪಡುತ್ತೇನೆ",
              "ميژھہ خود تے وؤننٹ کرنم",
              "म्हाग खाणे आवडत",
              "मैं खाना खाने के लाइ रे रस्ता है",
              "ഞാൻ ഭക്ഷണം കഴിക്കുന്നത് എനിക്ക് ഇഷ്ടമാണ്",
              "হম নোম হানাবা বাজেহান",
              "Dog कुत्ता কুকুৰ কुकुर कावादो",
              "કુતરો ನಾಯಿ کٹّا कुत्रो നായ থাবা",
              "Cat बिल्ली মানুহ বিড়াল",
              "मांजर बिरालो പൂച്ച পুই",
              "बियानो બિલાડી ಬೆಕ್ಕು بلّی",
              "Elephant हाथी হাতि हाति હાથી",
              "ಆನೆ ہاتھی हत्ती हाती ആന হाथি",
              "मी शाळेत जातो I am going to movie",
              "नमस्ते, तपाईँलाई कस्तो छ ଆକାଶ ନୀଳ ରଙ୍ଗର ",
              "ਬਚਾ ਕਿਤਾਬ ਪੜ੍ਹ ਰਿਹਾ ਹੈ मला मराठी येते",
              "अहम् छात्रः अस्मि ᱡᱚᱲᱟᱜ ᱯᱟᱨᱟᱢ ᱟᱢ।",
              "مون کي پاڻي کپي சிறந்த வார்த்தை",
              "నా పేరు మహేష్ एषः बालकः पठति",
              "میرا نام علی ہے तपाईंलाई कस्तो छ",
              "तुसीं चंगे हो మీరు ఎలా ఉన్నారు",
              "நான் செல்லும் ଆମେ ଘରେ ଯାଇଥାଏ",
              "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ کتاب میز پر है",
              "She is eating म खानु भयो",
              "हे घर माझं आहे ଏହାଁ ମୋ ଘର",
              "ਮੈਂ ਹੁਣ ਆ ਰਿਹਾ ਹਾਂ He is playing",
              "एतत् किं ମୁଁ ଓଡିଶାରେ ଯାଇଛି",
              "அவள் படிக்கிறாள் నేను తిన్నాను",
              "مان آيون ٿيندو آهيون ۾ सः पठति",
              "آپ کیسے ہیں؟ The sun is shining",
              "నేను వచ్చేస్తున్నాను ਉਸ ਨੇ ਪੜ੍ਹਾਈ ਕੀਤੀ ਹੈ",
              "ଏହାଁ କିଛି ଅଛି माता भोजनं पचति",
              "இது என் வீடு म नेपाल गएको छु",
              "माझं नाव जॉन आहे مان آيو ٿين ٿو",
              "یہ میرا گھر ہے। त्वं कस्य स्त्री असि",
              "అతను చదువుతున్నాడు This is my house",
              "तपाईंको नाम के हो ମୁଁ ଖାଇଥିବି",
              "ह्या आईचा आठवण वाटतं ఇది చిన్న పిల్లి",
              "அவள் சரி ਮੈਂ ਆਇਆ",
              "त्याचं आनंद आहे ସେ ସୁଖାନ୍ତି",
              "अहम् गच्छामि They are enjoying",
              "آپ کا سلامت رہنا। तत् पठति",
              "నేను ఉంటాను I came"
              ]})

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Snowball Stemmer for English
stemmer = SnowballStemmer("english")

# Preprocessing function
def preprocess(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove punctuation
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
    # Remove stopwords
    tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    return tokens

# Preprocess each sentence in the DataFrame
df['Preprocessed'] = df['Sentences'].apply(preprocess)


def predict_language(sentence):
    langs = set()  # Use a set to store unique languages
    for word in sentence.split():
        lang = langid.classify(word)[0]  # Predict language using langid library
        langs.add(lang)
    return list(langs)  # Convert the set to a list

# Apply language prediction to each sentence
df['Language_Prediction'] = df['Sentences'].apply(predict_language)

# Mapping of language codes to names
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


# Replace language codes with names in Language_Prediction column
df['Language_Prediction'] = df['Language_Prediction'].apply(lambda langs: list(set([language_names[lang] if lang in language_names else lang for lang in langs])))

# Save the DataFrame to a .pkl file
with open('language_predictions.pkl', 'wb') as file:
    pickle.dump(df, file)

# To load the DataFrame back, you can use the following code:
# with open('language_predictions.pkl', 'rb') as file:
#     loaded_df = pickle.load(file)
# print(loaded_df)
