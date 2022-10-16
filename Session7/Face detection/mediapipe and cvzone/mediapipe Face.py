import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
faceDetection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

cap=cv2.VideoCapture(0) #'folder/folder/test.avi'
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

count = 0

while cap.isOpened():
    ret,frame=cap.read()
    
    if ret==False :
        break
    frame_BGR=frame.copy()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(frame)
    rows, cols, _ = frame.shape
    
    if results.detections:
        for id , detection in enumerate(results.detections):
            #mp_drawing.draw_detection( frame, detection)
            
            bboxC=detection.location_data.relative_bounding_box
            bbox=int(bboxC.xmin*cols),int(bboxC.ymin*rows),\
            int(bboxC.width*cols),int(bboxC.height*rows)
            x,y,w,h = bbox
            ROI=frame_BGR[y:y+h,x:x+w]
            cv2.imwrite(f'faces/img{count}.jpg', ROI)
            count = count + 1
            cv2.rectangle(frame,bbox,(255,0,0),2)
            cv2.putText(frame,f'{int(detection.score[0]*100)} %',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    

    cv2.imshow('MediaPipe Faces', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()