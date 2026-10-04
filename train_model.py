import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/dna_dataset.csv")

X = df["sequence"]
y = df["family"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert DNA sequences into numerical features
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(
    max_iter=2000
)

model.fit(X_train_vec, y_train)

# Test model
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
print("=" * 50)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "models/dna_model.pkl")
joblib.dump(vectorizer, "models/dna_vectorizer.pkl")

print("\nMODEL SAVED SUCCESSFULLY!")