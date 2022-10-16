import numpy as np
import cv2
face_classifier = cv2.CascadeClassifier('HARR/haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0) #'folder/TEST.avi' , 'test.mp4'
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False :
        break
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,4)

    for (x,y,w,h) in faces:
        ROI=frame[y:y+h,x:x+w]
        ROI = cv2.resize(ROI,dsize=(224,224))
        
        
        cv2.imwrite(f'faces/img{count}.jpg', ROI)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        count = count + 1
        
    cv2.imshow('Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()