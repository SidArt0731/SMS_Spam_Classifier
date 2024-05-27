# SMS Spam Classifier

## Overview
This project implements a machine learning model to classify SMS messages as spam or ham (not spam). It utilizes natural language processing techniques and various machine learning algorithms to achieve accurate classification.

## Features
- Preprocessing of text data including removing punctuation, tokenization, and removing stop words.
- Training and evaluation of multiple machine learning models such as Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting, XGBoost, and LightGBM.
- Selection of the best performing model based on evaluation metrics like F1 score.
- Integration with a graphical user interface (GUI) for easy interaction and prediction.

## Comparison of Different Models Used

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.94     | 0.97      | 0.58   | 0.72     |
| Naive Bayes          | 0.96     | 1.00      | 0.73   | 0.84     |
| SVM                  | 0.96     | 0.99      | 0.75   | 0.85     |
| Random Forest        | 0.97     | 1.00      | 0.78   | 0.87     |
| Gradient Boosting    | 0.96     | 0.97      | 0.72   | 0.82     |
| XGBoost              | 0.97     | 0.95      | 0.83   | 0.88     |
| LightGBM             | 0.97     | 0.96      | 0.85   | 0.90     |

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
