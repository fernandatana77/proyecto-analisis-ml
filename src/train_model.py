import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from preprocessing import cargar_y_limpiar_datos


def entrenar_modelo():
  print("Cargando y preprocesando los datos...")
  df = cargar_y_limpiar_datos("data/amazon_reviews.csv")

  # Variables de entrada (X) y salida (y)
  X = df['clean_text']
  y = df['sentiment']

  # 1. Dividir los datos en entrenamiento (80%) y prueba (20%)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  print(
      f"Datos de entrenamiento: {len(X_train)} | Datos de prueba: {len(X_test)}"
  )

  # 2. Vectorización TF-IDF (transformar texto a matriz numérica)
  print("Vectorizando los textos con TF-IDF...")
  vectorizer = TfidfVectorizer(max_features=5000)
  X_train_vec = vectorizer.fit_transform(X_train)
  X_test_vec = vectorizer.transform(X_test)

  # 3. Entrenar el modelo de Regresión Logística
  print("Entrenando el modelo de Machine Learning...")
  model = LogisticRegression(max_iter=1000)
  model.fit(X_train_vec, y_train)

  # 4. Evaluar el modelo (Métricas requeridas por la rúbrica)
  print("\n--- MÉTRICAS DEL MODELO ---")
  y_pred = model.predict(X_test_vec)

  acc = accuracy_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred, average='weighted')

  print(f"Accuracy (Precisión global): {acc * 100:.2f}%")
  print(f"F1-Score: {f1 * 100:.2f}%")
  print("\nReporte de clasificación detallado:")
  print(classification_report(y_test, y_pred))

  # 5. Guardar el modelo y el vectorizador en la carpeta 'models/'
  print("Guardando el modelo y el vectorizador...")
  joblib.dump(model, 'models/sentiment_model.pkl')
  joblib.dump(vectorizer, 'models/vectorizer.pkl')
  print("¡Modelo guardado con éxito en la carpeta 'models/'!")


if __name__ == '__main__':
  entrenar_modelo()