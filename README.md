

# 📊 Détection de Maladies des Plantes avec ML 

---

## 1. Business Understanding
- **But** : Détecter automatiquement les maladies de plantes avec un modèle ML accessible via API et une interface Streamlit.
- **Objectif** : Aider les agriculteurs à identifier rapidement les maladies pour agir rapidement.

---

## 2. Data Understanding
- **Dataset utilisé** : [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Contenu** :  
  - 15 classes différentes de feuilles malades et saines
  - Exemples : *Tomato_Leaf_Mold*, *Potato_Late_blight*, *Tomato_healthy*.

---

## 3. Data Preparation

```mermaid
graph TD
    A[Data Preparation] --> B1[Resize 64x64]
    A --> B2[Grayscale Conversion]
    A --> B3[Normalization]
    A --> B4[Duplicate Removal]

```

- Redimensionnement des images (64x64)
- Conversion en niveaux de gris
- Normalisation des pixels
- Suppression des doublons

---

## 4. Modeling

```mermaid
flowchart TD
    A[Training Data] --> B[Model Selection]
    B --> C1[Random Forest]
    B --> C2["SVM (RBF Kernel)"]
    B --> C3[Decision Tree]
    B --> C4[KNN]
    C1 --> D[Evaluation]
    C2 --> D
    C3 --> D
    C4 --> D


```

- **Modèles entraînés** : Random Forest, SVM, Decision Tree, KNN.
- **Technologies** : Scikit-learn, NumPy, Joblib pour sauvegarde.
- **Splitting** : 80% entraînement, 20% test.

---

## 5. Evaluation

```mermaid
flowchart LR
    A[Metrics] --> B1(Accuracy)
    A --> B2(Precision)
    A --> B3(Recall)
    A --> B4(F1-Score)
    A --> B5(Confusion Matrix)
```

- **Résultats principaux** :
  - SVM ~ 64% Accuracy
  - Random Forest ~ 63% Accuracy
- **Analyse** :
  - SVM et RF plus robustes, KNN/DT sur-ajustement.

---

## 6. Deployment

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant API
    participant Model

    User->>Streamlit: Upload image & select model
    Streamlit->>API: POST /predict (image, model name)
    API->>Model: Load .pkl file
    Model-->>API: Prediction
    API-->>Streamlit: Return result + Recommendation
    Streamlit-->>User: Display disease & advice
```

- **API Flask** (`api.py`) : `/predict` endpoint
- **Interface Streamlit** (`app.py`) :
  - Upload d'image
  - Sélection du modèle
  - Affichage du résultat
- **Recommandations intégrées** selon la maladie détectée.

---

## 7. Future Work

```mermaid
flowchart TB
    A[Future Enhancements] --> B1[Image Segmentation]
    A --> B2[Cloud API Deployment]
    A --> B3[Support for other crops]
    A --> B4[Integration IoT + Serres]
```

- Segmentation d'images pour identifier la zone malade
- Déploiement sur Render/HuggingFace Spaces
- Extension aux cultures de blé, maïs, etc.
- Fusion avec capteurs dans des serres automatisées

---

## 8. Team and Acknowledgment
- **Étudiants** :
  - Binta Ball
  - Eya Zantour
- **Encadrant** : Khemais Abdallah
- **Année** : 2025, Projet de 4ᵉ année (Génie Informatique, spécialisation Data Science & IA)

---

