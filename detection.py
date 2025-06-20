import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt

# Configuration Roboflow
ROBOFLOW_API_KEY = "vV7dnPsE0Xx1xol0oZjx"
MODEL_ID = "apple-fruit-ajaxe-a17eg"
VERSION = "1"
URL = f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}?api_key={ROBOFLOW_API_KEY}"

# Charger l'image
image_path = "pomme2.jpg"
image = cv2.imread(image_path)

# Vérifier le chargement
if image is None:
    raise FileNotFoundError(f"Erreur : Fichier '{image_path}' introuvable. Vérifiez le chemin.")

# Préparer l'image pour l'API
_, img_encoded = cv2.imencode(".jpg", image)
files = {"file": img_encoded.tobytes()}

# Envoyer à Roboflow
try:
    response = requests.post(URL, files=files, timeout=10)
    response.raise_for_status()
    detections = response.json()
except Exception as e:
    raise ConnectionError(f"Erreur API Roboflow: {str(e)}")

# Vérifier les détections
if "predictions" not in detections:
    raise ValueError("Aucune détection trouvée dans la réponse")

# Dessiner les résultats
output_image = image.copy()
for detection in detections["predictions"]:
    x, y = detection["x"], detection["y"]
    w, h = detection["width"], detection["height"]

    # Coordonnées du rectangle
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)

    # Vérifier les limites de l'image
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Dessiner le rectangle et le texte
    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(output_image,
                f"{detection['class']} {detection['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (36, 255, 12), 2)

# Conversion pour Matplotlib
output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# Affichage avec Matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(output_rgb)
plt.axis('off')
plt.title('Détection de Pommes avec Roboflow/YOLOv8')
plt.show()

# Sauvegarde des résultats
cv2.imwrite("resultat_detection.jpg", output_image)
print("Résultats sauvegardés dans 'resultat_detection.jpg'")
