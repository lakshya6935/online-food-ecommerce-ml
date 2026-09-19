import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("ecommerce_delivery_analytics_cleaned.csv")

# Model 1: Delivery Delay
X = df[["Delivery Time (Minutes)"]]
y = df["Delivery Delay"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "delivery_delay_model.pkl")

# Model 2: Service Rating
X = df[["Customer Feedback"]]
y = df["Service Rating"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
prep = ColumnTransformer([
    ("feedback", OneHotEncoder(handle_unknown="ignore"), ["Customer Feedback"])
])
model2 = Pipeline([
    ("prep", prep),
    ("model", DecisionTreeClassifier(max_depth=20, random_state=42))
])
model2.fit(X_train, y_train)
joblib.dump(model2, "service_rating_model.pkl")

print("Both models trained and saved.")
