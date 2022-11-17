import numpy as np
import matplotlib.pyplot as plt
import cv2
from facenet_pytorch import MTCNN
import torch
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

device = torch.device(device)
mtcnn = MTCNN(image_size=160, margin=14, min_face_size=20, post_process=False)
###############################################
cap=cv2.VideoCapture('testr4.avi')
###############################################
classes = {'Abdallah': 0, 'Alshimaa': 1,'Hossam':2,'Khaled':3,'Shima':4,'Zain':5}
def ImageClass(n):
    for x , y in classes.items():
        if n == y :
            return x
size = (1600, 1200)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation

def vgg_face():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model
model = vgg_face()

model.load_weights('vgg_face_weights.h5')
from joblib import dump, load
scaler=load( 'scaler.joblib') 
pca=load( 'pca_model.joblib') 
clf=load( 'SVC.joblib') 
from tensorflow.keras.models import Model
model = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

from numpy import expand_dims
from cv2 import resize,INTER_CUBIC
from tensorflow.keras.preprocessing.image import  img_to_array

def preprocess_image(img):
    img = img_to_array(img)
    img = img/255.0
    img = expand_dims(img, axis=0)
    return img

def Face_Recognition(roi,model,scaler,pca,clf):
    roi=resize(roi,dsize=(224,224),interpolation=INTER_CUBIC)
    roi=preprocess_image(roi)
    embedding_vector = model.predict(roi)[0]

    embedding_vector=scaler.transform(embedding_vector.reshape(1, -1))
    embedding_vector_pca = pca.transform(embedding_vector)
    # result1 = clf.predict(embedding_vector_pca)[0]
    #print(result1)
    y_predict = clf.predict_proba(embedding_vector_pca)[0]
    #print(y_predict)
    
    result = np.where(y_predict > 0.4)[0]
    
    return result

result_video = cv2.VideoWriter('testr44.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
font = cv2.FONT_HERSHEY_SIMPLEX     
fontScale = 1
color = (255, 0, 0)
thickness = 2
while True :
    ret, frame = cap.read()
    if not ret:
        break  
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame=cv2.resize(frame,(1600,1200),interpolation=cv2.INTER_CUBIC)
    frame=cv2.GaussianBlur(frame, ksize=(3,3), sigmaX=0)
    boxes, probs = mtcnn.detect(frame, landmarks=False)
    
    
    if  not probs.all() == None and probs.all()>0.95 :
            for x1,y1,x2,y2 in boxes :
                x1,x2,y1,y2=int(x1),int(x2),int(y1),int(y2)
                roi=frame[y1:y2,x1:x2]
                img_size=cv2.resize(roi,(224,224),interpolation=cv2.INTER_CUBIC)
                result=Face_Recognition(roi,model,scaler,pca,clf)
                if len(result) > 1 :
                    cv2.putText(frame, ImageClass(result[0]) , (x1-5,y1-5), font,fontScale, color, thickness, cv2.LINE_AA)
                elif  len(result)== 0 :
                    #print('other')
                    cv2.putText(frame, 'Other' , (x1-5,y1-5), font,fontScale, color, thickness, cv2.LINE_AA)
                else :
                    cv2.putText(frame, ImageClass(result) , (x1-5,y1-5), font,fontScale, color, thickness, cv2.LINE_AA)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)             
    result_video.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
result_video.release()
cv2.destroyAllWindows() 