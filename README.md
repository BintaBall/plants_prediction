

# 📈 Détection de Maladies des Plantes avec ML + API + Streamlit — *Approche CRISP-DM*

---

## 1. Business Understanding
- **Objectif**: Créer une application accessible capable de détecter automatiquement les maladies de plantes à partir d’images de feuilles.
- **Motivation**: Améliorer la détection rapide des maladies pour soutenir les agriculteurs, en particulier dans les environnements à faibles ressources.
- **Livrables**: API Flask + Interface utilisateur Streamlit avec prédiction de maladies et recommandations agricoles.

---

## 2. Data Understanding
- **Source**: [Dataset PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease).
- **Contenu**:
  - Plus de 50,000 images de feuilles de plantes, classifiées en 15 classes (maladies et feuilles saines).
  - Exemple de classes : *Tomato_Bacterial_spot*, *Potato_Early_blight*, *Tomato_healthy*, etc.
- **Observation**: Présence de déséquilibre de classes (certains types de maladies sont surreprésentés).

---

## 3. Data Preparation
- **Étapes**:
  - Redimensionnement des images.
  - Conversion en niveaux de gris.
  - Normalisation des pixels.
  - Suppression des doublons (via perceptual hashing).
- **Structure**:
  ```
  ├── models/ (modèles sauvegardés)
  ├── train_models/ (prétraitement, entraînement)
  ├── results/ (matrices de confusion, courbes d'évaluation)
  ```

---

## 4. Modeling
- **Modèles utilisés**:
  - Random Forest
  - Support Vector Machine (SVM RBF kernel)
  - Decision Tree
  - K-Nearest Neighbors (KNN)
- **Techniques**:
  - Split du dataset: 80% entraînement / 20% test.
  - Outils: Scikit-learn, NumPy.
- **Sauvegarde**: Modèles exportés sous format `.pkl`.

---

## 5. Evaluation
- **Indicateurs utilisés**:
  - Accuracy
  - F1-Score
  - Matrices de confusion
- **Résultats**:
  - Visualisation via `accuracy_plot.png` et `f1_scores_plot.png`.
  - Analyse détaillée des erreurs par modèle.
- **Observations**:
  - Random Forest et SVM ont montré de meilleures performances que KNN et Decision Tree.

---

## 6. Deployment
- **API**: 
  - Développement d'une API REST avec Flask (`api.py`).
- **Interface Utilisateur**:
  - Déploiement local avec Streamlit (`app_streamlit.py`).
- **Utilisation**:
  1. Choix du modèle.
  2. Upload d'une image.
  3. Affichage du résultat et d'une **recommandation agricole** adaptée.
- **Exemples de recommandations**:
  - *Tomato__Tomato_mosaic_virus* → "Enlever immédiatement la plante affectée."
  - *Potato___Early_blight* → "Appliquer un fongicide."

---

## 7. Future Work
- Ajouter la segmentation d'images pour localiser les maladies.
- Déployer l'API sur le cloud (Render, HuggingFace Spaces...).
- Étendre la couverture à d’autres cultures comme le maïs ou le blé.
- Fusionner le projet avec des capteurs en serre (IoT + Computer Vision).

---

## 8. Auteurs
- **Contexte**: Projet académique 4ᵉ année, Génie Informatique — Data Science & IA, 2025.
- **Participants**:
  - **Eya Zantour**
  - **Binta Ball**
- **Encadrant**: Khemais Abdallah

