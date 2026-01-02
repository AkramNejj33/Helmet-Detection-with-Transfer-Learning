import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PARAMÈTRES
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32

# ============================================================
# CHARGER LE MODÈLE
# ============================================================
print("📁 Chargement du modèle final...")
model = keras.models.load_model('model_final.h5')
print("✓ Modèle chargé")

# ============================================================
# CHARGER LES DONNÉES DE TEST
# ============================================================
print("\n📁 Chargement des données de test...")

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    'data/dataset/test',
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
print(f"\nMatrice de confusion :\n{cm}")

# Rapport détaillé
print("\n" + classification_report(
    y_true, y_pred_classes,
    target_names=['No Helmet', 'Helmet']
))

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
plt.savefig('confusion_matrix.png', dpi=100)
plt.show()

print("\n✅ Évaluation terminée !")
print("📊 Matrice : confusion_matrix.png")