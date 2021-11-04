from cv2 import cv2

# img = cv2.imread('images/h4.jpg')
# img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# neural = cv2.CascadeClassifier('frontalface_default.xml')
#
# # Предположим что наша сетка натренированна на маленьких изображениях scaleFactor говорит о множителе размера "лиц"
# results = neural.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
#
# for (x, y, w, h) in results:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), thickness=3)
#
# cv2.imshow("result", img)
# cv2.waitKey(0)
cap = cv2.VideoCapture(0)
cap.set(4, 500)
cap.set(3, 300)

while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    neural = cv2.CascadeClassifier('frontalface_default.xml')

    results = neural.detectMultiScale(gray, scaleFactor=1.2111, minNeighbors=3)
    for (x, y, w, h) in results:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
    cv2.imshow("VideoCaption", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
