import cv2

img = cv2.imread('Artifact1-Tests/Test3.jpg') #reading image
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) #resizing for certain photos

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
trained_cascade = cv2.CascadeClassifier('cascades/cascade.xml')

rectangles = trained_cascade.detectMultiScale(gray_img, 1.3, 5) #scalefact=1.3 minNeighbors=5

for (x, y, width, height) in rectangles:
    cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 5)  # drawing the rectangle for faces

cv2.imshow('Detected faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()