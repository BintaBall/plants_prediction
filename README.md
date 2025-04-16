```markdown
# Plants Disease Prediction

Ce projet a pour but de détecter automatiquement des maladies de plantes à partir d’images de feuilles. Il combine deux modèles de Machine Learning (Random Forest et SVM), une API Flask pour faire les prédictions, et une interface utilisateur avec Streamlit.

---

## Structure du projet

```
plants_prediction/
├── PlantVillage/           # Dossier contenant les images classées par maladie
├── train_models.py         # Script d'entraînement des modèles ML
├── api.py                  # API Flask pour la prédiction
├── app.py                  # Interface Streamlit pour l'utilisateur
├── requirements.txt        # Liste des dépendances Python
```

---

## Description technique

- **Dataset utilisé** : [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Nombre de classes** : 16 maladies ou états de santé différents de plantes
- **Prétraitement** : Redimensionnement des images à 64x64 pixels, conversion en niveaux de gris
- **Modèles utilisés** :
  - Random Forest Classifier
  - Support Vector Machine (SVM, kernel RBF)
- **Interface** : Streamlit (upload d’image, choix du modèle, affichage des prédictions)
- **API** : Flask REST API locale, réception d’image et réponse en JSON

---

## Installation

### 1. Créer un environnement virtuel (optionnel mais recommandé)

```bash
python -m venv venv
# Linux / macOS :
source venv/bin/activate
# Windows :
venv\Scripts\activate
```

### 2. Installer les dépendances

Avec `requirements.txt` :

```bash
pip install -r requirements.txt
```

Sinon :

```bash
pip install numpy pandas pillow scikit-learn joblib flask streamlit requests
```

---

## Entraînement des modèles

Avant de prédire, il faut entraîner les modèles sur les données PlantVillage :

```bash
python train_models.py
```

Ce script sauvegarde deux fichiers :
- `random_forest.pkl`
- `svm_rbf.pkl`

Ces fichiers sont chargés automatiquement par l’API.

---

## Lancer l’API Flask

Dans un terminal :

```bash
python api.py
```

Cela démarre un serveur local accessible à l'adresse :  
`http://127.0.0.1:5000/predict`

---

## Lancer l'application Streamlit

Dans un autre terminal :

```bash
streamlit run app.py
```

L'application web s'ouvre dans le navigateur.  
Elle permet :
- de télécharger une image de feuille
- de choisir le modèle de prédiction
- d’obtenir le résultat instantanément

---

## Exemple d’appel API (optionnel)

- **Méthode** : POST  
- **URL** : `http://127.0.0.1:5000/predict`  
- **Données** : `multipart/form-data`
  - `file` : image (jpg/png)
  - `model` : `"Random Forest"` ou `"SVM RBF"`

**Réponse JSON** :

```json
{
  "prediction": 5
}
```

(La prédiction est l'index de la classe prédite.)

---

## Fonctionnalités futures

- Remplacement par des modèles Deep Learning (CNN)
- Affichage des noms de classes dans l’API
- Déploiement cloud (Render, Hugging Face, Heroku, etc.)
- Ajout de métriques de performance dans l’interface

---

## Auteurs & Contexte

Projet académique – 4ᵉ année en Génie Informatique, spécialisation Data Science & IA  
Année : 2025  
Encadré dans le cadre du module Machine Learning
Encadrant: Khemais Abdallah
Etudiant: Binta Ball DS4
Etudiant: Aya Zantour DS4
```

---

Souhaite-tu que je te crée un vrai fichier `requirements.txt` adapté à ce projet aussi ?