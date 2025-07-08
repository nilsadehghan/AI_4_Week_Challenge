import string
import pandas as pd
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, classification_report

# (Optional) Set custom NLTK data path if needed
nltk.data.path.append(r"C:\Users\Lenovo\AppData\Roaming\nltk_data")



# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load the dataset
df = pd.read_csv("sentiment_dataset_en_large.csv")

# Check label distribution
print(df['label'].value_counts())

# Text preprocessing function
def filter(text):
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_words = []
    for w in tokens:
        if w not in stop_words and w not in string.punctuation:
            lemma = lemmatizer.lemmatize(w)
            filtered_words.append(lemma)
    return " ".join(filtered_words)

# Apply preprocessing
df['text_filtered'] = df['text'].apply(filter)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['text_filtered'])
y = df['label']

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate the model
precision = precision_score(Y_test, y_pred, average='macro')
recall = recall_score(Y_test, y_pred, average='macro')

# Print results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

joblib.dump(model,"sentiment_model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")