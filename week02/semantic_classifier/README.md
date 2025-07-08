# Sentiment Analysis API with FastAPI

This project is a simple English text sentiment analysis system consisting of two main parts:

1. *Model Training:  
   - Uses labeled text data from `sentiment_dataset_en_large.csv`  
   - Text preprocessing (tokenization, stopwords removal, lemmatization)  
   - Text vectorization using TF-IDF  
   - Training a Logistic Regression model  
   - Evaluating the model and saving it with `joblib`  

2. Web API with FastAPI*:  
   - Loading the saved model and vectorizer  
   - Defining an endpoint to receive text input and predict sentiment  
   - Providing a REST API for integration with other applications or services  

---

## Project Structure

/ ├── sentiment_dataset_en_large.csv  # Dataset with texts and labels ├── sentiment_model.pkl             # Trained model saved file ├── vectorizer.pkl                 # Saved TF-IDF vectorizer ├── main.py                       # FastAPI app with API endpoints └── train.py                      # Script to train and save the model

---

## Requirements

- Python 3.7+  
- Packages: `fastapi`, `uvicorn`, `scikit-learn`, `pandas`, `nltk`, `joblib`

Install dependencies via:

```bash
pip install fastapi uvicorn scikit-learn pandas nltk joblib


---

Setup Instructions

1. Download NLTK data (if not already done):



import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

2. Prepare your dataset: Make sure sentiment_dataset_en_large.csv is placed in the project root.


3. Train the model:



python train.py

This will preprocess the data, train the Logistic Regression model, and save sentiment_model.pkl and vectorizer.pkl.

4. Run the FastAPI server:



uvicorn main:app --reload

The API will be accessible at http://127.0.0.1:8000.


---

API Usage

Endpoint: /predict/ (POST)

Request JSON example:


{
  "text": "I really love this product!"
}

Response example:


{
  "sentiment": "positive"
}


---

Notes

The dataset size and quality affect the model accuracy. Consider improving the dataset or model for better results.

The preprocessing steps include lowercasing, tokenization, stopword removal, and lemmatization to normalize text input.

The API uses a Logistic Regression classifier and TF-IDF vectorizer saved with joblib.



---

License

This project is open-source and free to use.


---

Feel free to open issues or contribute to improve the project!