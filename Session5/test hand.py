import cv2
from RaiseHand import detect_hand

cap=cv2.VideoCapture('Test Project_Trim.avi')
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
##########################################
size = (1600, 1200)
result_video = cv2.VideoWriter('handdetect11.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
##########################################

while cap.isOpened() :
    ret, frame = cap.read()
    if not ret:
        break
    Hand=0
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_hand=cv2.resize(frame,(640,640),interpolation=cv2.INTER_CUBIC)
    #raise Hand
    res=detect_hand(frame_hand)
    for i in range (len(res)) :
        xmin,ymin,xmax,ymax,confidence,_,_=res.iloc[i,:]
        xmin,ymin,xmax,ymax=int(xmin),int(ymin),int(xmax),int(ymax)
        if confidence > 0.4 :
            cv2.rectangle(frame, (xmin * 1600 // 640, ymin * 1200  // 640), (xmax * 1600 // 640, ymax * 1200  // 640), (255, 0, 0), 2)
            Hand = Hand + 1
    
    frame[:70,:,:]=-10      
    cv2.putText(frame, str('Raise Hand      '+str(Hand)), (10,40), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA) 
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    frame=cv2.resize(frame,(1600,1200),interpolation=cv2.INTER_CUBIC)
    result_video.write(frame)
    cv2.imshow("Project", frame)
    
    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
        break
cap.release()
result_video.release()
cv2.destroyAllWindows()