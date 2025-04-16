import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

#  Dossier contenant les sous-dossiers (classes)
data_dir = 'C:/Users/hp/Dropbox/PC/Desktop/plants_prediction/PlantVillage'

# 🔍 Récupération automatique de toutes les classes (sous-dossiers)
categories = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
])
print("Catégories utilisées :", categories)

# Création du dictionnaire de labels
labels = {category: idx for idx, category in enumerate(categories)}

X = []
y = []

# Chargement et traitement des images
for category in categories:
    path = os.path.join(data_dir, category)
    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).convert('L')  # Niveaux de gris
            img = img.resize((64, 64))               # Redimensionnement
            img_array = np.array(img).flatten()      # Transformation en vecteur
            X.append(img_array)
            y.append(labels[category])
        except Exception as e:
            print(f"Erreur avec {img_name} dans {category}: {e}")
            continue  # Ignorer les images corrompues

# Conversion en numpy arrays
X = np.array(X) / 255.0  # Normalisation
y = np.array(y)

print(f"Nombre total d'images : {len(X)}")
print(f"Nombre total de classes : {len(categories)}")

# Division en données d'entraînement et de test
if len(X) > 0 and len(y) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.8, random_state=42)
else:
    raise ValueError("Les données sont vides. Vérifiez les images.")

#  Modèles à entraîner
classifiers = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    SVC(kernel='rbf', gamma='scale'),
]

model_names = ['random_forest', 'svm_rbf']

# Entraînement, évaluation, sauvegarde
for clf, name in zip(classifiers, model_names):
    print(f"\nEntraînement du modèle : {name}")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {acc:.2f}')
    joblib.dump(clf, f'{name}.pkl')

print("\n✅ Tous les modèles ont été entraînés et sauvegardés avec succès.")
