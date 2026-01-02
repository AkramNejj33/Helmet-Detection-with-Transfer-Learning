"""
Serveur Flask pour l'interface de prédiction du modèle Helmet Detection
Lance avec: python Scripts/predict_server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
from pathlib import Path

# ============================================================
# CONFIGURATION ET CHEMINS
# ============================================================
app = Flask(__name__)
CORS(app)  # Autoriser les requêtes cross-origin

# Obtenir les chemins (Scripts est le dossier parent du projet)
SCRIPT_DIR = Path(__file__).parent  # Scripts/
PROJECT_ROOT = SCRIPT_DIR.parent     # Racine du projet

MODEL_PATH = str(PROJECT_ROOT / 'models' / 'model_final.h5')

IMG_SIZE = 224
CLASS_NAMES = ['No Helmet', 'Helmet']

# ============================================================
# CHARGER LE MODÈLE
# ============================================================
print("🔄 Chargement du modèle...")

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Modèle chargé : {MODEL_PATH}")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    MODEL_LOADED = False
    model = None

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def prepare_image(image_bytes):
    """
    Préparer une image pour la prédiction
    
    Args:
        image_bytes : image en bytes
    
    Returns:
        array : image préparée de shape (1, 224, 224, 3)
    """
    try:
        # Ouvrir l'image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convertir en RGB (au cas où ce soit RGBA ou grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionner
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convertir en array numpy
        img_array = np.array(img) / 255.0
        
        # Ajouter dimension batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"Erreur lors du traitement de l'image : {e}")

def predict_helmet(img_array):
    """
    Faire une prédiction sur une image
    
    Args:
        img_array : image préparée
    
    Returns:
        dict : résultats de la prédiction
    """
    if not MODEL_LOADED or model is None:
        raise RuntimeError("Le modèle n'est pas chargé")
    
    # Prédire
    predictions = model.predict(img_array, verbose=0)
    
    # Obtenir la classe et la confiance
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])
    class_name = CLASS_NAMES[class_idx]
    
    return {
        'class': class_name,
        'confidence': confidence,
        'probabilities': {
            'No Helmet': float(predictions[0][0]),
            'Helmet': float(predictions[0][1])
        }
    }

# ============================================================
# ROUTES
# ============================================================

@app.route('/', methods=['GET'])
def home():
    """Route de base"""
    return jsonify({
        'status': 'ok',
        'message': 'Serveur Helmet Detection prêt',
        'model_loaded': MODEL_LOADED,
        'endpoints': {
            'POST /predict': 'Prédire sur une image',
            'GET /health': 'Vérifier la santé du serveur'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Vérifier la santé du serveur"""
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL_LOADED,
        'model_path': MODEL_PATH
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction
    
    Accepte une image en POST et retourne la prédiction
    
    Returns:
        JSON : {
            'class': 'Helmet' ou 'No Helmet',
            'confidence': 0.95,
            'probabilities': {'No Helmet': 0.05, 'Helmet': 0.95}
        }
    """
    
    # Vérifier que le modèle est chargé
    if not MODEL_LOADED or model is None:
        return jsonify({
            'error': 'Le modèle n\'est pas chargé',
            'status': 'error'
        }), 500
    
    # Vérifier qu'une image a été uploadée
    if 'image' not in request.files:
        return jsonify({
            'error': 'Aucune image fournie. Utilisez la clé "image"',
            'status': 'error'
        }), 400
    
    file = request.files['image']
    
    # Vérifier que le fichier n'est pas vide
    if file.filename == '':
        return jsonify({
            'error': 'Aucun fichier sélectionné',
            'status': 'error'
        }), 400
    
    # Vérifier le format
    allowed_formats = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_formats:
        return jsonify({
            'error': 'Format non supporté. Utilisez JPG, PNG, GIF ou BMP',
            'status': 'error'
        }), 400
    
    try:
        # Lire le fichier
        image_bytes = file.read()
        
        # Préparer l'image
        img_array = prepare_image(image_bytes)
        
        # Faire la prédiction
        result = predict_helmet(img_array)
        
        # Ajouter le statut
        result['status'] = 'success'
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400
    
    except Exception as e:
        return jsonify({
            'error': f'Erreur lors de la prédiction : {str(e)}',
            'status': 'error'
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Endpoint pour prédictions batch (multiple images)
    
    Accepte plusieurs images et retourne les prédictions
    """
    
    if not MODEL_LOADED or model is None:
        return jsonify({
            'error': 'Le modèle n\'est pas chargé',
            'status': 'error'
        }), 500
    
    # Vérifier qu'au moins une image a été uploadée
    if 'images' not in request.files:
        return jsonify({
            'error': 'Aucune image fournie. Utilisez la clé "images"',
            'status': 'error'
        }), 400
    
    files = request.files.getlist('images')
    
    if len(files) == 0:
        return jsonify({
            'error': 'Aucun fichier sélectionné',
            'status': 'error'
        }), 400
    
    results = []
    
    for file in files:
        try:
            if file.filename == '':
                continue
            
            # Lire et traiter l'image
            image_bytes = file.read()
            img_array = prepare_image(image_bytes)
            
            # Prédire
            result = predict_helmet(img_array)
            result['filename'] = file.filename
            result['status'] = 'success'
            
            results.append(result)
        
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
                'status': 'error'
            })
    
    return jsonify({
        'predictions': results,
        'total': len(results),
        'status': 'success'
    }), 200

# ============================================================
# GESTION DES ERREURS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Route non trouvée',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Erreur interne du serveur',
        'status': 'error'
    }), 500

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Serveur Helmet Detection en cours de démarrage...")
    print("="*60)
    print(f"📁 Modèle : {MODEL_PATH}")
    print(f"✅ Modèle chargé : {MODEL_LOADED}")
    print("="*60)
    print("🌐 Serveur accessible sur : http://localhost:5000")
    print("="*60)
    print("\n📚 Endpoints disponibles :")
    print("  GET  /                 - Infos du serveur")
    print("  GET  /health          - Santé du serveur")
    print("  POST /predict         - Prédiction sur une image")
    print("  POST /predict/batch   - Prédictions sur plusieurs images")
    print("\n" + "="*60 + "\n")
    
    # Lancer le serveur
    app.run(debug=True, host='0.0.0.0', port=5000)