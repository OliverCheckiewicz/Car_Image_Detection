 import cv2
import matplotlib
from matplotlib import pyplot as plt

car_cascade = cv2.CascadeClassifier('/Users/moj_synio_kochany/Desktop/cars.xml')
img = cv2.imread('/Users/moj_synio_kochany/Desktop/car3.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect cars
cars = car_cascade.detectMultiScale(gray, 1.1, 1)

# Draw border
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    cars = cars + 1

# Show image
plt.figure(figsize=(10,20))
plt.imshow(img)
