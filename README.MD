# SMS Spam Classifier

## Overview
This project implements a machine learning model to classify SMS messages as spam or ham (not spam). It utilizes natural language processing techniques and various machine learning algorithms to achieve accurate classification.

## Features
- Preprocessing of text data including removing punctuation, tokenization, and removing stop words.
- Training and evaluation of multiple machine learning models such as Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting, XGBoost, and LightGBM.
- Selection of the best performing model based on evaluation metrics like F1 score.
- Integration with a graphical user interface (GUI) for easy interaction and prediction.

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
