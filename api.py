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
decision_tree = joblib.load('decision_tree.pkl')
knn = joblib.load('knn.pkl')

# Récupération des classes depuis les sous-dossiers
categories = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
])
label_to_class = {idx: category for idx, category in enumerate(categories)}

# Dictionnaire de recommandations (exemple à compléter)
recommendations = {
    'Pepper__bell___Bacterial_spot': "Supprimez les feuilles infectées. Évitez l'excès d'humidité. Utilisez un fongicide si nécessaire.",
    'Pepper__bell___healthy': "Aucune action nécessaire. Continuez à surveiller régulièrement.",
    'Potato___Early_blight': "Retirez les feuilles infectées et appliquez un fongicide à base de cuivre.",
    'Potato___Late_blight': "Détruisez les plantes touchées. Ne pas composter. Utilisez des variétés résistantes.",
    'Potato___healthy': "Pas de problème détecté. Maintenez de bonnes pratiques culturales.",
    'Tomato_Bacterial_spot': "Évitez l’arrosage par aspersion. Retirez les plantes très touchées.",
    'Tomato_Early_blight': "Retirez les feuilles infectées. Utilisez du compost mûr. Appliquez un fongicide naturel.",
    'Tomato_Late_blight': "Retirez immédiatement les plantes. Désinfectez les outils après usage.",
    'Tomato_Leaf_Mold': "Améliorez la ventilation. Appliquez un traitement fongique biologique.",
    'Tomato_Septoria_leaf_spot': "Retirez les feuilles basses. Arrosez à la base uniquement.",
    'Tomato_Spider_mites_Two_spotted_spider_mite': "Pulvérisez un acaricide biologique. Augmentez l'humidité ambiante.",
    'Tomato__Target_Spot': "Éliminez les feuilles atteintes. Favorisez la rotation des cultures.",
    'Tomato__Tomato_YellowLeaf__Curl_Virus': "Arrachez la plante. Contrôlez les aleurodes (mouches blanches).",
    'Tomato__Tomato_mosaic_virus': "Arrachez les plantes malades. Nettoyez les outils. Évitez les semences non certifiées.",
    'Tomato_healthy': "Plante saine. Continuez à bien l'entretenir."
}


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
    elif model_name == "Decision Tree":
        prediction = decision_tree.predict(image_array)
    elif model_name == "KNN":
        prediction = knn.predict(image_array)
    else:
        return jsonify({"error": "Nom de modèle invalide. Choisir entre 'Random Forest', 'SVM RBF', 'Decision Tree', ou 'KNN'."}), 400

    predicted_label = int(prediction[0])
    predicted_class = label_to_class.get(predicted_label, "Classe inconnue")
    recommendation = recommendations.get(predicted_class, "Aucune recommandation disponible.")

    return jsonify({
    "prediction_index": predicted_label,
    "prediction_class": predicted_class,
    "recommendation": recommendation
    })


if __name__ == '__main__':
    app.run(debug=True)
