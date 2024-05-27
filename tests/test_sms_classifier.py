import sys
import os
import unittest
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sms_classifier import SMSClassifier

# Set up logging
logging.basicConfig(filename='test_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class TestSMSClassifier(unittest.TestCase):
    def setUp(self):
        config = {
            'data_path': 'SMSSpamCollection.tsv',
            'test_size': 0.2
        }
        self.classifier = SMSClassifier(config)
        # Expanded dataset for testing
        self.df = pd.DataFrame({
            'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
            'body_text': [
                'Hello, how are you?',
                'Win a free iPhone now!',
                'Are you coming to the party?',
                'Congratulations! You have won a lottery!',
                'Let\'s have a meeting tomorrow.',
                'Get your free entry to the event!',
                'Can you send me the report?',
                'You have been selected for a prize!',
                'Don\'t forget to bring the documents.',
                'Claim your reward by clicking here!'
            ]
        })
        self.df = self.classifier.preprocess_data(self.df)
        self.X = self.classifier.vectorizer.fit_transform(self.df['body_text_clean'])
        self.y = self.df['label'].map({'ham': 0, 'spam': 1})
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=config['test_size'], random_state=42
        )
        logging.info("Setup complete.")

    def test_model_training(self):
        self.classifier.model = LGBMClassifier(min_data_in_leaf=1, min_data_in_bin=1)
        self.classifier.model.fit(self.X_train, self.y_train)
        y_pred = self.classifier.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        logging.info(f"Model training accuracy: {accuracy:.2f}")
        self.assertGreater(accuracy, 0.5)

    def test_model_evaluation(self):
        self.classifier.model = LGBMClassifier(min_data_in_leaf=1, min_data_in_bin=1)
        accuracy, precision, recall, f1, _ = self.classifier.evaluate_model(
            self.classifier.model, self.X_train, self.X_test, self.y_train, self.y_test
        )
        logging.info(f"Model evaluation - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
        self.assertGreater(accuracy, 0.5)
        self.assertGreater(precision, 0.5)
        self.assertGreater(recall, 0.5)
        self.assertGreater(f1, 0.5)

    def test_prediction(self):
        self.classifier.model = LGBMClassifier(min_data_in_leaf=1, min_data_in_bin=1)
        self.classifier.model.fit(self.X, self.y)
        test_messages = [
            'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C\'s apply 08452810075over18',
            'Nah I don\'t think he goes to usf, he lives around here though'
        ]
        predictions = self.classifier.predict(test_messages)
        logging.info(f"Test predictions: {predictions}")
        self.assertEqual(len(predictions), 2)

if __name__ == '__main__':
    unittest.main()
