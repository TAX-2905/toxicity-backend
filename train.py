import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load dataset
data = pd.read_csv("data/kreol_toxicity.csv")

texts = data["text"].astype(str)
labels = data["label"].map({"toxic": 1, "non_toxic": 0})

# 2. Preprocess (minimal)
def preprocess(text):
    return text.lower().strip()

texts = texts.apply(preprocess)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# 4. Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 6. Evaluation
y_pred = model.predict(X_test_vec)

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# 7. Save model and vectorizer
joblib.dump(model, "model/toxicity_model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")

print("\n✅ Model and vectorizer saved successfully")
