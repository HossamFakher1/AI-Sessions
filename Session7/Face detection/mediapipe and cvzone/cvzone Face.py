from cvzone.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

detector = FaceDetector()
count=0
while cap.isOpened():
    ret,frame=cap.read()
    if ret==False :
        break
    frame_BGR=frame.copy()
    
    frame, bboxs = detector.findFaces(frame)

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        for bbox in bboxs :
            x,y,w,h = bbox["bbox"]
            ROI=frame_BGR[y:y+h,x:x+w]
            cv2.imwrite(f'faces/img{count}.jpg', ROI)
            count = count + 1

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()