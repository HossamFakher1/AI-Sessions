import dlib
import cv2

detector =dlib.get_frontal_face_detector()

cap=cv2.VideoCapture(0) # 'folder/folder/test.avi'
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

count = 0

while cap.isOpened() :
    
    ret ,frame =cap.read()
    #frame=cv2.resize(frame,dsize=(1000,700))
    if ret == False :
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= detector(gray, 2)
    
    for face in faces :
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        ROI=frame[y1:y2,x1:x2]
        cv2.imwrite(f'faces/img{count}.jpg', ROI)
        count = count + 1
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()