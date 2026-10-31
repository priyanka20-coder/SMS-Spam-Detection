from flask import Flask, render_template, request
import os, zipfile, urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

app = Flask(__name__)
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
os.makedirs("data", exist_ok=True)
ZIP_PATH = "data/sms_spam_collection.zip"
DATA_PATH = "data/SMSSpamCollection"

def train_model():
    if not os.path.exists(DATA_PATH):
        if not os.path.exists(ZIP_PATH):
            urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall("data")

    df = pd.read_csv(DATA_PATH, sep="\\t", header=None,
                     names=["label", "message"], encoding="latin-1")
    df = df.dropna()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna()
    df["label"] = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.20,
        random_state=42, stratify=df["label"])

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
        "precision": round(precision_score(y_test, predictions, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, predictions, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, predictions, zero_division=0) * 100, 2),
        "total": len(df),
        "spam": int((df["label"] == 1).sum()),
        "ham": int((df["label"] == 0).sum())
    }
    return vectorizer, model, metrics, confusion_matrix(y_test, predictions), classification_report(
        y_test, predictions, target_names=["Not Spam", "Spam"], zero_division=0)

try:
    vectorizer, model, metrics, cm, report = train_model()
    model_error = None
except Exception as e:
    vectorizer = model = metrics = cm = report = None
    model_error = str(e)

@app.route("/", methods=["GET", "POST"])
def home():
    result = score = message = None
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if not message:
            result = "Please enter a message."
        elif model is None:
            result = "Model could not load. Check internet connection and restart."
        else:
            transformed = vectorizer.transform([message])
            prediction = int(model.predict(transformed)[0])
            score = round(float(model.predict_proba(transformed)[0][1]) * 100, 2)
            result = "SPAM" if prediction == 1 else "NOT SPAM"

    return render_template("index.html", result=result, score=score,
                           message=message, metrics=metrics, cm=cm,
                           report=report, model_error=model_error)

if __name__ == "__main__":
    app.run(debug=True)
