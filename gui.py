import os
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sms_classifier import SMSClassifier

class SMSClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SMS Spam Classifier")
        self.root.geometry("400x300")

        self.classifier = SMSClassifier({'data_path': 'data/SMSSpamCollection.tsv', 'test_size': 0.2})
        self.model_path = 'model/classifier.joblib'
        self.vectorizer_path = 'model/vectorizer.joblib'

        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        self.label = tk.Label(self.root, text="Enter SMS:", font=("Arial", 12))
        self.label.pack(pady=10)

        self.text_entry = tk.Text(self.root, height=5, width=40)
        self.text_entry.pack(pady=10)

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_sms)
        self.predict_button.pack(pady=5)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

        self.save_button = tk.Button(self.root, text="Save Data", command=self.save_data)
        self.save_button.pack(pady=5)

    def load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.classifier.load_model(self.model_path, self.vectorizer_path)
        else:
            self.classifier.run()
            self.classifier.save_model(self.model_path, self.vectorizer_path)

    def predict_sms(self):
        sms = self.text_entry.get("1.0", tk.END).strip()
        if sms:
            prediction = self.classifier.predict([sms])
            result = "Spam" if prediction[0] else "Ham"
            self.result_label.config(text=f"Prediction: {result}")
            self.text_entry.delete("1.0", tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter an SMS.")

    def save_data(self):
        sms = self.text_entry.get("1.0", tk.END).strip()
        prediction = self.result_label.cget("text").split(": ")[1].lower()
        if sms and prediction:
            label = 'spam' if prediction == 'Spam' else 'ham'
            new_data = pd.DataFrame({'label': [label], 'body_text': [sms]})
            data_path = self.classifier.config['data_path']
            if os.path.exists(data_path):
                existing_data = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'body_text'])
                new_data = pd.concat([existing_data, new_data])
            new_data.to_csv(data_path, sep='\t', index=False, header=False)
            messagebox.showinfo("Success", "Data saved successfully. Retraining the model...")
            self.classifier.run()
            self.classifier.save_model(self.model_path, self.vectorizer_path)
        else:
            messagebox.showwarning("Save Error", "No data to save or prediction missing.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SMSClassifierApp(root)
    root.mainloop()