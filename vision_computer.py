import cv2
from ultralytics import YOLO

# Charge le modèle YOLOv8 pré-entraîné (COCO)
model = YOLO("yolov8n.pt")  # ou ton propre modèle si tu as entraîné sur "apple" et "person"

# Ouvre la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
    exit()

print("Appuie sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur de capture vidéo")
        break

    # Prédiction YOLOv8
    results = model.predict(frame, conf=0.5)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box.astype(int)
            class_id = int(cls)
            class_name = model.model.names[class_id]  # nom de la classe

            # Couleur différente selon la classe
            if class_name.lower() == "person":
                color = (255, 0, 0)  # Bleu pour humain
            elif class_name.lower() == "apple":
                color = (0, 255, 0)  # Vert pour pomme
            else:
                color = (0, 255, 255)  # Jaune pour autres objets

            # Dessine la boîte et le label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Détection (personne et pomme)", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
