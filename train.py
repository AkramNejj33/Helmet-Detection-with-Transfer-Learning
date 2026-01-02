import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PARAMÈTRES
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10

# ============================================================
# ÉTAPE 1 : CHARGER LES DONNÉES
# ============================================================
print("📁 Chargement des données...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'data/dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_test_datagen.flow_from_directory(
    'data/dataset/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print(f"✓ Train samples : {train_gen.samples}")
print(f"✓ Val samples : {val_gen.samples}")

# ============================================================
# ÉTAPE 2 : CRÉER LE MODÈLE
# ============================================================
print("\n🧠 Création du modèle...")

# Charger MobileNetV2 pré-entraîné
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Geler toutes les couches
base_model.trainable = False

# Ajouter les nouvelles couches
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs, outputs)

print(f"✓ Modèle créé avec {model.count_params()} paramètres")

# ============================================================
# ÉTAPE 3 : COMPILER
# ============================================================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Modèle compilé")

# ============================================================
# ÉTAPE 4 : ENTRAÎNEMENT PHASE 1 (Feature Extraction)
# ============================================================
print("\n" + "="*60)
print("PHASE 1 : FEATURE EXTRACTION (Couches gelées)")
print("="*60)

history_phase1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=val_gen.samples // BATCH_SIZE,
    verbose=1
)

model.save('model_phase1.h5')
print("✓ Modèle Phase 1 sauvegardé")

# ============================================================
# ÉTAPE 5 : ENTRAÎNEMENT PHASE 2 (Fine-Tuning)
# ============================================================
print("\n" + "="*60)
print("PHASE 2 : FINE-TUNING (Dégel partiel)")
print("="*60)

# Dégeler les 30 dernières couches
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompiler avec learning rate plus faible
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_phase2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE2,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=val_gen.samples // BATCH_SIZE,
    verbose=1
)

model.save('model_final.h5')
print("✓ Modèle final sauvegardé")

# ============================================================
# ÉTAPE 6 : VISUALISATION DES COURBES
# ============================================================
print("\n📊 Génération des courbes...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history_phase1.history['loss'], label='Phase 1 - Train Loss')
axes[0].plot(history_phase1.history['val_loss'], label='Phase 1 - Val Loss')
axes[0].plot(
    [EPOCHS_PHASE1 + i for i in range(EPOCHS_PHASE2)],
    history_phase2.history['loss'],
    label='Phase 2 - Train Loss'
)
axes[0].plot(
    [EPOCHS_PHASE1 + i for i in range(EPOCHS_PHASE2)],
    history_phase2.history['val_loss'],
    label='Phase 2 - Val Loss'
)
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].set_title('Courbes de Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy
axes[1].plot(history_phase1.history['accuracy'], label='Phase 1 - Train Acc')
axes[1].plot(history_phase1.history['val_accuracy'], label='Phase 1 - Val Acc')
axes[1].plot(
    [EPOCHS_PHASE1 + i for i in range(EPOCHS_PHASE2)],
    history_phase2.history['accuracy'],
    label='Phase 2 - Train Acc'
)
axes[1].plot(
    [EPOCHS_PHASE1 + i for i in range(EPOCHS_PHASE2)],
    history_phase2.history['val_accuracy'],
    label='Phase 2 - Val Acc'
)
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Courbes d\'Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=100)
plt.show()

print("\n✅ Entraînement terminé !")
print("📁 Modèle final : model_final.h5")
print("📊 Courbes : training_curves.png")