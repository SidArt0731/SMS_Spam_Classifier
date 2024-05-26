import os
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import string
import re
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure nltk data is downloaded
nltk.download('stopwords', quiet=True)

class SMSClassifier:
    def __init__(self, config):
        self.config = config
        self.model = LGBMClassifier()
        self.vectorizer = TfidfVectorizer()

    def load_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r') as file:
            raw_data = file.read()

        parsed_data = raw_data.replace("\t", "\n").split("\n")
        labels = parsed_data[0::2]
        texts = parsed_data[1::2]

        if len(labels) != len(texts):
            min_length = min(len(labels), len(texts))
            labels = labels[:min_length]
            texts = texts[:min_length]
            logging.warning("Mismatch between labels and texts length. Truncated to minimum length.")

        return pd.DataFrame({'label': labels, 'body_text': texts})

    def preprocess_text(self, text: str) -> str:
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = re.split(r'\W+', text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word and word not in stop_words]
        return ' '.join(tokens)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['body_text_clean'] = df['body_text'].apply(self.preprocess_text)
        return df

    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        return accuracy, precision, recall, f1, conf_mat

    def run(self):
        # Load and preprocess data
        df = self.load_data(self.config['data_path'])
        df = self.preprocess_data(df)

        # Feature extraction
        X = self.vectorizer.fit_transform(df['body_text_clean'])
        y = df['label'].map({'ham': 0, 'spam': 1})

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['test_size'], random_state=42)

        # Evaluate the model
        accuracy, precision, recall, f1, conf_mat = self.evaluate_model(self.model, X_train, X_test, y_train, y_test)
        logging.info(f"LightGBM - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

        # Train the model on the entire dataset
        self.model.fit(X, y)

    def save_model(self, model_path: str, vectorizer_path: str):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        logging.info("Model and vectorizer saved")

    def load_model(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        logging.info("Model and vectorizer loaded")

    def predict(self, messages: list) -> list:
        processed_messages = [self.preprocess_text(msg) for msg in messages]
        features = self.vectorizer.transform(processed_messages)
        return self.model.predict(features)
    

if __name__ == "__main__":
    config = {
        'data_path': 'data/SMSSpamCollection.tsv',
        'test_size': 0.2
    }

    sms_classifier = SMSClassifier(config)
    sms_classifier.run()
    sms_classifier.save_model('sms_classifier_model.pkl', 'sms_vectorizer.pkl')

