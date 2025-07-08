from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.data.path.append(r"C:\Users\Lenovo\AppData\Roaming\nltk_data")

# Load saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define input data format
class InputText(BaseModel):
    text: str

stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    return " ".join(filtered)

# Define a POST route
@app.post("/predict/")

def predict_sentiment(data: InputText):

    """
    Takes input text and returns the sentiment prediction.
    -text:The text string to analyze
    -A JASON with the predicted sentiment label.
    """
    filtered_text = preprocess(data.text)
    vec = vectorizer.transform([filtered_text])
    prediction = model.predict(vec)[0]
    return {"sentiment": prediction}


