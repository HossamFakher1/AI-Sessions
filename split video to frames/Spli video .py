import cv2
i=0
count=0
cap=cv2.VideoCapture('Hossam.mp4') # path of video
print(cap.isOpened())
while cap.isOpened() :
    
    ret,frame =cap.read()
    if ret == False :
        break

    #frame=cv2.resize(frame,(1280,1280),interpolation=cv2.INTER_CUBIC)
    
    if count % 8 == 0 :
        cv2.imwrite(f'dataset/hossam/{i}1hossam.png',frame)
        i = i + 1
    count = count + 1

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


