import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def setup_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean(doc):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    doc = doc.lower()
    doc = re.sub(r'[^a-z\s]', ' ', doc)

    tokens = nltk.word_tokenize(doc)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)
