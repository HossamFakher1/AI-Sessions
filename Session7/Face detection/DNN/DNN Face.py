import cv2
DNN = "TF"
if DNN == "CAFFE":
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

cap=cv2.VideoCapture(0) # 'folder/folder/test.avi'
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

count = 0

while cap.isOpened() :
    ret , frame=cap.read()
    frame_BGR=frame.copy()
    if ret == False :
        break
    frameHeight , frameWidth =frame.shape[:2]


    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 177, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            ROI=frame_BGR[y1:y2,x1:x2]
            cv2.imwrite(f'faces/img{count}.jpg', ROI)
            count = count + 1
            cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),3)
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    cv2.imshow('frame',frame)
        
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()