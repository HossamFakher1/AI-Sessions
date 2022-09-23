import cv2
from torch.hub import load   
# Model
model = load('yolov5', 'custom','yolov5/knifebest.pt', source='local')



cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)
##########################################
size = (640, 640)
result_video = cv2.VideoWriter('handdetect1121.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
##########################################

while cap.isOpened() :
    ret, frame = cap.read()
    if not ret:
        break
    
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    frame=cv2.resize(frame,(640,640),interpolation=cv2.INTER_CUBIC)
    
    results = model(frame, size=640)  
    res=results.pandas().xyxy[0]
    for i in range (len(res)) :
        xmin,ymin,xmax,ymax,confidence,_,_=res.iloc[i,:]
        xmin,ymin,xmax,ymax=int(xmin),int(ymin),int(xmax),int(ymax)
        if confidence > 0.6 :
            cv2.rectangle(frame, (xmin , ymin), (xmax , ymax), (255, 0, 0), 2)
            cv2.putText(frame, str(res.name[i]), (xmin , ymin), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA) 
            cv2.putText(frame, str(round(confidence,2)), (xmin+50 , ymin), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA) 
           

    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    frame=cv2.resize(frame,(640,640),interpolation=cv2.INTER_CUBIC)
    result_video.write(frame)
    cv2.imshow("Project", frame)
    
    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
        break
cap.release()
result_video.release()
cv2.destroyAllWindows()