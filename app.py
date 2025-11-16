from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo y scaler
model = tf.keras.models.load_model("modelo_rendimiento.keras")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "API de Predicción funcionando correctamente"

@app.route("/predecir", methods=["POST"])
def predecir():
    datos = request.get_json()

    # Convertir en array
    X = np.array([[ 
        datos["responsabilidad"], 
        datos["notas"], 
        datos["situacion"], 
        datos["motivacion"], 
        datos["nivel"] 
    ]])

    # Escalar
    X_scaled = scaler.transform(X)

    # Predicción
    prediccion = float(model.predict(X_scaled)[0][0])

    return jsonify({"prediccion": prediccion})

if __name__ == "__main__":
    app.run()
