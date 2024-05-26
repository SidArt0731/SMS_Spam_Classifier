import os
import unittest
import pandas as pd
from sms_spam_classifier.sms_classifier import SMSClassifier

class TestSMSClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = {
            'data_path': 'test_data.tsv',
            'test_size': 0.2
        }
        cls.test_data_content = (
            "ham\tI'm going to home now.\n"
            "spam\tFree entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.\n"
            "ham\tSee you tomorrow!\n"
            "spam\tCongratulations! You've won a free ticket.\n"
        )
        with open(cls.config['data_path'], 'w') as f:
            f.write(cls.test_data_content)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.config['data_path'])

    def setUp(self):
        self.classifier = SMSClassifier(self.config)
    
    def test_load_data(self):
        df = self.classifier.load_data(self.config['data_path'])
        self.assertEqual(len(df), 4)
        self.assertEqual(list(df.columns), ['label', 'body_text'])

    def test_preprocess_text(self):
        text = "Congratulations! You've won a free ticket."
        expected = "congratulations won free ticket"
        processed_text = self.classifier.preprocess_text(text)
        self.assertEqual(processed_text, expected)

    def test_preprocess_data(self):
        df = self.classifier.load_data(self.config['data_path'])
        df_clean = self.classifier.preprocess_data(df)
        self.assertIn('body_text_clean', df_clean.columns)
        self.assertEqual(df_clean['body_text_clean'].iloc[1], "free entry 2 wkly comp win fa cup final tkts 21st may 2005")

    def test_train_and_evaluate(self):
        df = self.classifier.load_data(self.config['data_path'])
        df = self.classifier.preprocess_data(df)
        X = self.classifier.vectorizer.fit_transform(df['body_text_clean'])
        y = df['label'].map({'ham': 0, 'spam': 1})
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['test_size'], random_state=42)
        self.classifier.train(X_train, y_train)
        accuracy, precision, recall, f1, conf_mat = self.classifier.evaluate(X_test, y_test)
        
        self.assertGreater(accuracy, 0)
        self.assertGreater(precision, 0)
        self.assertGreater(recall, 0)
        self.assertGreater(f1, 0)
        self.assertEqual(conf_mat.shape, (2, 2))

    def test_predict(self):
        df = self.classifier.load_data(self.config['data_path'])
        df = self.classifier.preprocess_data(df)
        X = self.classifier.vectorizer.fit_transform(df['body_text_clean'])
        y = df['label'].map({'ham': 0, 'spam': 1})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['test_size'], random_state=42)
        self.classifier.train(X_train, y_train)
        
        test_messages = [
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
            "I'm going to home now."
        ]
        predictions = self.classifier.predict(test_messages)
        self.assertEqual(len(predictions), 2)
        self.assertIn(1, predictions)  # Spam
        self.assertIn(0, predictions)  # Ham

if __name__ == '__main__':
    unittest.main()