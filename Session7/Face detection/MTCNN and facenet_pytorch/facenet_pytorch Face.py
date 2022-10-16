# libraries
import numpy as np
from facenet_pytorch import MTCNN
import cv2
mtcnn = MTCNN(image_size=160, margin=14, min_face_size=20,device='cpu', post_process=False)

cap = cv2.VideoCapture(0) #Test Project.avi 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

count = 0
while cap.isOpened() :

    ret, frame = cap.read()
    if not ret:
        break
    frame_BGR=frame.copy()
    
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(frame, landmarks=False)
    
    if  not probs.all() == None and probs.all() > 0.6 :
        for x1,y1,x2,y2 in boxes :
            x1,x2,y1,y2=int(x1),int(x2),int(y1),int(y2)
            ROI=frame_BGR[y1:y2,x1:x2]
            cv2.imwrite(f'faces/img{count}.jpg', ROI)
            count = count + 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    cv2.imshow("Project", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # if press q
        break

cap.release()
cv2.destroyAllWindows()