import streamlit as st
from PIL import Image
import requests
import numpy as np
import io

# Titre de l'application
st.title("Prédiction de Maladies des Plantes")

# URL de l'API Flask
API_URL = "http://127.0.0.1:5000/predict"

# Sélection du modèle
model_name = st.selectbox(
    "Sélectionnez un modèle",
    ["Random Forest", "SVM RBF", "Decision Tree", "KNN"]
)

# Téléchargement de l'image
uploaded_file = st.file_uploader("Téléchargez une image de feuille (format JPG/PNG)", type=["png", "jpg", "jpeg"])

# Affichage de l'image
if uploaded_file is not None:
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert('L').resize((64, 64))

    st.image(image, caption="Image téléchargée", use_column_width=False, width=200)

# Bouton prédiction
if st.button("Prédire"):
    if uploaded_file is not None:
        uploaded_file.seek(0)
        files = {"file": uploaded_file}
        data = {"model": model_name}

        with st.spinner("Analyse en cours..."):
            response = requests.post(API_URL, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            prediction_class = result.get("prediction_class", "Inconnue")
            prediction_index = result.get("prediction_index", -1)
            recommendation = result.get("recommendation", "Aucune recommandation disponible.")

            st.success(f"Résultat : {prediction_class} (Index : {prediction_index})")
            st.info(f"**Recommandation :** {recommendation}")
        else:
            st.error(f"Erreur de l'API : {response.text}")
    else:
        st.warning("Veuillez d'abord téléverser une image.")
