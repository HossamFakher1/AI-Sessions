from torch.hub import load   
from cv2 import resize ,INTER_CUBIC
# Model
model = load('yolov5', 'custom','yolov5/best.pt', source='local')


def detect_hand(frame) :
    frame=resize(frame,dsize=(640,640),interpolation=INTER_CUBIC)
    results = model(frame, size=640)  
    res=results.pandas().xyxy[0]
    return res