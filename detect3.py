import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO("yolov8n.pt")  # Замените модель, обученную на яблоках

# Функция предварительной обработки изображения
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Преобразование HSV
    image = cv2.GaussianBlur(image, (5, 5), 0)      # Réduction du bruit
    return image

#  Загрузите два изображения, имитирующие стерео (левое и правое изображения)
img_left = cv2.imread('pomme2.jpg')
img_right = cv2.imread('pomme2.jpg')


# Предварительная обработка
img_left_processed = preprocess_image(img_left)
img_right_processed = preprocess_image(img_right)

# Обнаружение яблок с помощью YOLOv8 на левом изображении
results = model.predict(img_left, conf=0.5)
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_left, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_left, "Apple", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# 3D-реконструкция по стереоизображениям (упрощенно)
# Преобразование в оттенки серого
grayL = cv2.cvtColor(img_left_processed, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(img_right_processed, cv2.COLOR_BGR2GRAY)

# StereoBM для карты диспаратности
stereo = cv2.StereoBM_create(numDisparities=16*3, blockSize=15)
disparity = stereo.compute(grayL, grayR)

# Image avec détection
plt.figure(figsize=(10,5))
plt.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
plt.title("Left image - Detection")
plt.axis('off')
plt.show()

# Carte de disparité
plt.figure(figsize=(10,5))
plt.imshow(disparity, cmap='gray')
plt.title("Disparity map")
plt.axis('off')
plt.show()

#cv2.imwrite("Left image - Detection", img_left)
#cv2.imwrite("Disparity map", cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX))
#print("Images saved successfully")
cv2.waitKey(0)
cv2.destroyAllWindows()