import cv2
from ultralytics import YOLO
import rospy

# Chargement du modèle YOLOv8
model = YOLO("yolov8n.pt")

# Capture d'image
image = cv2.imread("pommier.jpg")

# Prédiction
results = model.predict(image)

# Décision : si pomme mûre détectée, action du bras robotisé
if results.success:
    rospy.publish("/robot_arm/move", "cueillir")
