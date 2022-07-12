import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #getting the face cascade
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') #getting the eye cascade

while True:
    ret, frame = cap.read() #get cam capture

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale
    faces = face_cascade.detectMultiScale(img, 1.3, 5) #scalefact=1.3 minNeighbors=5
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 5) #drawing the rectangle for faces
        roi_gray = img[y:y+width, x:x+width] #rows -> columns
        roi_color = frame[y:y+height, x:x+width]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5) #scalefact=1.3 minNeighbors=5
        for (ex, ey, ewidth, eheight) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ewidth, ey + eheight), (0, 255, 0), 5) #drawing the rectangle for eyes

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('w'): #wait till 'w' is pressed
        break

cap.release()
cv2.destroyAllWindows()