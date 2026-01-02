import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# ============================================================
# DÉFINIR LES CHEMINS (Scripts est dans le dossier parent du projet)
# ============================================================
SCRIPT_DIR = Path(__file__).parent  # Scripts/
PROJECT_ROOT = SCRIPT_DIR.parent     # Racine du projet (parent de Scripts)

# Créer les chemins
DATA_DIR = PROJECT_ROOT / 'data' / 'dataset'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# ============================================================
# CRÉER LES DOSSIERS S'ILS N'EXISTENT PAS
# ============================================================
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("✓ Dossiers 'models' et 'results' créés/vérifiés")
print(f"📁 Répertoire racine : {PROJECT_ROOT}")
print(f"📁 Données : {DATA_DIR}")
print(f"📁 Modèles : {MODELS_DIR}")
print(f"📁 Résultats : {RESULTS_DIR}\n")

# ============================================================
# PARAMÈTRES
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32

# ============================================================
# CHARGER LE MODÈLE
# ============================================================
print("📁 Chargement du modèle final...")

model_path = MODELS_DIR / 'model_final.h5'

if not model_path.exists():
    print(f"❌ Erreur : Le fichier {model_path} n'existe pas")
    print("   Lance d'abord : python Scripts/train.py")
    exit(1)

model = keras.models.load_model(str(model_path))
print(f"✓ Modèle chargé : {model_path}")

# ============================================================
# CHARGER LES DONNÉES DE TEST
# ============================================================
print("\n📁 Chargement des données de test...")

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    str(DATA_DIR / 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"✓ Test samples : {test_gen.samples}")

# ============================================================
# ÉVALUATION
# ============================================================
print("\n" + "="*60)
print("ÉVALUATION SUR LE TEST SET")
print("="*60)

# Prédictions
y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

# Accuracy
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"\n✓ Test Accuracy : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"✓ Test Loss : {test_loss:.4f}")

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred_classes)
print(f"\nMatrice de confusion :")
print(cm)

# Rapport détaillé
classification_rep = classification_report(
    y_true, y_pred_classes,
    target_names=['No Helmet', 'Helmet']
)

print("\n" + classification_rep)

# ============================================================
# SAUVEGARDER LES RÉSULTATS DANS UN FICHIER
# ============================================================
print("\n📝 Sauvegarde des résultats...")

results_file = RESULTS_DIR / 'evaluation_results.txt'

with open(results_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("RÉSULTATS D'ÉVALUATION - HELMET DETECTION\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Test Accuracy : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
    f.write(f"Test Loss : {test_loss:.4f}\n\n")
    
    f.write("Matrice de Confusion :\n")
    f.write(str(cm) + "\n\n")
    
    f.write("Classification Report :\n")
    f.write(classification_rep)

print(f"✓ Résultats sauvegardés : {results_file}")

# ============================================================
# VISUALISATION : MATRICE DE CONFUSION
# ============================================================
print("\n📊 Génération de la matrice de confusion...")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Helmet', 'Helmet'],
            yticklabels=['No Helmet', 'Helmet'])
plt.title('Matrice de Confusion')
plt.ylabel('Vrai Label')
plt.xlabel('Prédiction')
plt.tight_layout()

# Sauvegarder la matrice
matrix_path_results = RESULTS_DIR / 'confusion_matrix.png'

plt.savefig(str(matrix_path_results), dpi=100)
plt.show()

print(f"✓ Matrice sauvegardée : {matrix_path_results}")

print("\n✅ Évaluation terminée !")
print(f"📁 Modèle utilisé : {model_path}")
print(f"📊 Matrice : {matrix_path_results}")
print(f"📝 Résultats : {results_file}")