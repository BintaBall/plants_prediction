from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Dossier des images d'entraînement (sert pour récupérer les classes)
data_dir = 'C:/Users/hp/Dropbox/PC/Desktop/plants_prediction/PlantVillage'

# Charger les modèles
random_forest = joblib.load('random_forest.pkl')
svm_rbf = joblib.load('svm_rbf.pkl')

# Récupération des classes depuis les sous-dossiers
categories = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
])
label_to_class = {idx: category for idx, category in enumerate(categories)}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier envoyé"}), 400

    file = request.files['file']
    file.seek(0)

    try:
        image = Image.open(io.BytesIO(file.read())).convert('L').resize((64, 64))
        image_array = np.array(image).flatten().reshape(1, -1) / 255.0
    except Exception as e:
        return jsonify({"error": f"Erreur de traitement d'image: {str(e)}"}), 400

    model_name = request.form.get('model')
    if model_name == "Random Forest":
        prediction = random_forest.predict(image_array)
    elif model_name == "SVM RBF":
        prediction = svm_rbf.predict(image_array)
    else:
        return jsonify({"error": "Nom de modèle invalide. Choisir entre 'Random Forest' ou 'SVM RBF'."}), 400

    predicted_label = int(prediction[0])
    predicted_class = label_to_class.get(predicted_label, "Classe inconnue")

    return jsonify({
        "prediction_index": predicted_label,
        "prediction_class": predicted_class
    })

if __name__ == '__main__':
    app.run(debug=True)
