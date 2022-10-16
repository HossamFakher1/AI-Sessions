import cv2
from mtcnn.mtcnn import MTCNN

detector=MTCNN()

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

count = 0
while cap.isOpened() :
    
    ret,frame =cap.read()
    if ret == False :
        break
    frame_BGR=frame.copy()
 
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    detections=detector.detect_faces(frame)
    for det in detections:
        if det['confidence'] >= 0.7:
            x, y, w, h = det['box']
            ROI=frame_BGR[y:y+h,x:x+w]
            cv2.imwrite(f'faces/img{count}.jpg', ROI)
            count = count + 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    cv2.imshow('frame',frame)
        
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()