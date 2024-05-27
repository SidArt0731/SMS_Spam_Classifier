# SMS Spam Classifier

## Overview
This project implements a machine learning model to classify SMS messages as spam or ham (not spam). It utilizes natural language processing techniques and various machine learning algorithms to achieve accurate classification.

## Features
- Preprocessing of text data including removing punctuation, tokenization, and removing stop words.
- Training and evaluation of multiple machine learning models such as Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting, XGBoost, and LightGBM.
- Selection of the best performing model based on evaluation metrics like F1 score.
- Integration with a graphical user interface (GUI) for easy interaction and prediction.

## Comparison of Different Models Used

- Logistic Regression - Accuracy: 0.94, Precision: 0.97, Recall: 0.58, F1 Score: 0.72
- Naive Bayes - Accuracy: 0.96, Precision: 1.00, Recall: 0.73, F1 Score: 0.84
- SVM - Accuracy: 0.96, Precision: 0.99, Recall: 0.75, F1 Score: 0.85
- Random Forest - Accuracy: 0.97, Precision: 1.00, Recall: 0.78, F1 Score: 0.87
- Gradient Boosting - Accuracy: 0.96, Precision: 0.97, Recall: 0.72, F1 Score: 0.82
- XGBoost - Accuracy: 0.97, Precision: 0.95, Recall: 0.83, F1 Score: 0.88
- LightGBM - Accuracy: 0.97, Precision: 0.96, Recall: 0.85, F1 Score: 0.90

## Dependencies
This project relies on the following Python libraries:

- pandas
- nltk
- scikit-learn
- lightgbm
- joblib

These dependencies can be installed using pip:
```bash
pip install pandas nltk scikit-learn lightgbm joblib
```

## Usage
- Clone the repository:
```bash
git clone https://github.com/SidArt0731/SMS_Spam_Classifier.git
```
- Set up a virtual environment (optional but recommended):
```bash
python -m venv venv
```

- Activate the virtual environment:

On Windows:
```bash
venv\Scripts\activate
```
On macOS/Linux::
```bash
source venv/bin/activate
```

- Install the project dependencies:
```bash
pip install -r requirements.txt
```

- Run the SMS Spam Classifier GUI:
```bash
python gui.py
```

## Author
SidData

## License
This project is licensed under the MIT License.
