import cv2
import requests
import matplotlib.pyplot as plt

# 🔹 Récupérer le modèle
ROBOFLOW_API_KEY = "vV7dnPsE0Xx1xol0oZjx"
MODEL_ID = "apple-fruit-ajaxe-a17eg"
VERSION = "1"

# 🔹 URL du modèle
url = f"https://detect.roboflow.com/{MODEL_ID}/{VERSION}?api_key={ROBOFLOW_API_KEY}"

# 🔹 Charger une image
image_path = "pomme.jpg"
image = cv2.imread(image_path)

# Vérifier si l'image est bien chargée
if image is None:
    raise ValueError("Erreur : Impossible de charger l'image. Vérifie le chemin du fichier.")

# 🔹 Convertir en format utilisable pour l'API
_, img_encoded = cv2.imencode(".jpg", image)
files = {"file": img_encoded.tobytes()}

# 🔹 Envoyer à Roboflow et récupérer les résultats
response = requests.post(url, files=files)
detections = response.json()
print(detections)

# 🔹 Dessiner les boîtes de détection et les labels
for detection in detections["predictions"]:
    x, y, width, height = detection["x"], detection["y"], detection["width"], detection["height"]
    label = detection["class"]
    confidence = detection["confidence"]

    cv2.rectangle(image,
                  (int(x - width // 2), int(y - height // 2)),
                  (int(x + width // 2), int(y + height // 2)),
                  (0, 255, 0), 2)

    cv2.putText(image, f"{label} ({confidence:.2f})", (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 🔹 Solution 1 - Affichage des résultats avec Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,6))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Détection des pommes")
plt.show()

# 🔹 Solution 2 - Enregistrement correct de l’image
cv2.imwrite("detection_result.jpg", image)
print("✅ Image enregistrée sous 'detection_result.jpg', ouvre-la pour voir les résultats.")

# 🔹 Solution 3 - Graphique de précision des détections
confidences = [d["confidence"] for d in detections["predictions"]]
labels = [d["class"] for d in detections["predictions"]]

plt.figure(figsize=(8,5))
plt.bar(range(len(confidences)), confidences, color="green")
plt.xticks(range(len(confidences)), labels, rotation=45)
plt.xlabel("Détection")
plt.ylabel("Confiance (%)")
plt.title("Précision des détections des pommes")
plt.show()
