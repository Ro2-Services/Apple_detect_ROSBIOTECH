import cv2
import requests
import numpy as np

# Configuration Roboflow
ROBOFLOW_API_KEY = "vV7dnPsE0Xx1xol0oZjx"
MODEL_ID = "apple-fruit-ajaxe-a17eg"
VERSION = "1"
URL = f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}?api_key={ROBOFLOW_API_KEY}"

# Ouvre la webcam (0 = caméra par défaut)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

print("Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de capture vidéo")
        break

    # Préparer l'image pour Roboflow
    _, img_encoded = cv2.imencode(".jpg", frame)
    files = {"file": img_encoded.tobytes()}

    try:
        response = requests.post(URL, files=files, timeout=10)
        response.raise_for_status()
        detections = response.json()
    except Exception as e:
        print(f"Erreur API Roboflow: {str(e)}")
        continue

    # Dessiner les détections
    for detection in detections.get("predictions", []):
        x, y = detection["x"], detection["y"]
        w, h = detection["width"], detection["height"]
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{detection['class']} {detection['confidence']:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

    # Afficher le résultat
    cv2.imshow("Détection de pommes (Roboflow)", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
