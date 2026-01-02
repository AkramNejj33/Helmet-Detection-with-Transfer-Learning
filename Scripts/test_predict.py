"""
Script simple pour tester le serveur de prédiction
Utilise: python test_predict.py
"""

import requests
import sys
from pathlib import Path

# URL du serveur
SERVER_URL = 'http://localhost:5000'

def test_image(image_path):
    """Tester une prédiction sur une image"""
    
    if not Path(image_path).exists():
        print(f"❌ Fichier non trouvé : {image_path}")
        return
    
    try:
        # Ouvrir le fichier image
        with open(image_path, 'rb') as f:
            files = {'image': f}
            
            # Faire la requête POST
            response = requests.post(
                f'{SERVER_URL}/predict',
                files=files
            )
        
        # Afficher le résultat
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Prédiction réussie !")
            print(f"📸 Image : {image_path}")
            print(f"🎯 Classe : {data['class']}")
            print(f"📊 Confiance : {data['confidence']:.2%}")
            print(f"📈 Probabilités :")
            print(f"   - No Helmet : {data['probabilities']['No Helmet']:.2%}")
            print(f"   - Helmet : {data['probabilities']['Helmet']:.2%}")
        else:
            print(f"❌ Erreur {response.status_code} : {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Erreur : Impossible de se connecter au serveur")
        print("   Assure-toi que le serveur est lancé avec : python predict_server.py")
    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == '__main__':
    # Utilisation : python test_predict.py path/to/image.jpg
    if len(sys.argv) < 2:
        print("Usage : python test_predict.py <chemin_vers_image>")
        print("Exemple : python test_predict.py test.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_image(image_path)