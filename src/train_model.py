import joblib
import pandas as pd
from preprocessing import cargar_y_limpiar_datos
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


def entrenar_modelo():
  print("Cargando y preprocesando los datos...")
  df = cargar_y_limpiar_datos("data/amazon_reviews.csv")

  X = df["clean_text"]
  y = df["sentiment"]

  # Dividir datos
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  print(
      f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}"
  )

  # Vectorización TF-IDF optimizada para capturar mejor palabras clave de sentimiento
  print("Vectorizando textos con TF-IDF...")
  vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
  X_train_vec = vectorizer.fit_transform(X_train)
  X_test_vec = vectorizer.transform(X_test)

  # Entrenar con class_weight='balanced' para corregir el sesgo hacia los positivos
  print(
      "Entrenando modelo de Regresión Logística (con balanceo de clases)..."
  )
  model = LogisticRegression(max_iter=1000, class_weight="balanced")
  model.fit(X_train_vec, y_train)

  # Evaluar métricas
  y_pred = model.predict(X_test_vec)
  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, average="weighted")

  print(f"\n--- MÉTRICAS DEL NUEVO MODELO ---")
  print(f"Accuracy: {acc * 100:.2f}%")
  print(f"F1-Score: {f1 * 100:.2f}%")
  print(classification_report(y_test, y_pred))

  # Guardar archivos
  joblib.dump(model, "models/sentiment_model.pkl")
  joblib.dump(vectorizer, "models/vectorizer.pkl")
  print("¡Modelo re-entrenado y guardado con éxito!")


if __name__ == "__main__":
  entrenar_modelo()