import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort import *

video_path = r'D:\Download\DTP project\Vehicle-Counting-in-Lanes\vehicle2.mp4'
vid = cv2.VideoCapture(video_path)
model = YOLO('yolov8n.pt' )

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

laneA = np.array([[616,499], [951, 490], [936, 715], [506, 686], [616,499]], np.int32)
laneB = np.array([[961,503], [1303,488], [1438,720], [950,712], [961,503]], np.int32)

laneA_Line = np.array([laneA[0],laneA[1]]).reshape(-1)
laneB_Line = np.array([laneB[0],laneB[1]]).reshape(-1)

tracker = Sort()
laneAcount = []
laneBcount = []

while True:
    ret, frm = vid.read()
    frm = cv2.resize(frm, (1920,1080))
    rslts = model(frm)
    current_detections = np.empty([0,5])

    for inf in rslts:
        parameters = inf.boxes
        for box in parameters:
            X1, Y1, X2, Y2 = box.xyxy[0]
            X1, Y1, X2, Y2 = int(X1), int(Y1), int(X2), int(Y2)
            h,w =  Y2 - Y1,X2 - X1
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)
            cvzone.putTextRect(frm, f'{class_detect}', [X1 + 8, Y1 - 12], thickness=2, scale=1)
            cv2.rectangle(frm, (X1, Y1), (X2, Y2), (0, 255, 0), 2)

            if class_detect == 'truck' or class_detect == 'car' or class_detect == 'bus'\
                    and conf > 60:
                detections = np.array([X1,Y1,X2,Y2,conf])
                current_detections = np.vstack([current_detections,detections])

    cv2.polylines(frm,[laneA], isClosed=False, color=(66, 84, 245), thickness=6)
    cv2.polylines(frm, [laneB], isClosed=False, color=(84, 245, 66), thickness=6)

    track_results = tracker.update(current_detections)
    for result in track_results:
        X1,Y1,X2,Y2,id = result
        X1,Y1,X2,Y2,id = int(X1),int(Y1),int(X2),int(Y2),int(id)
        h,w=  Y2 - Y1,X2 - X1
        cx, cy = X1 + w // 2, Y1 + h // 2 -40


        if laneA_Line[0] < cx < laneA_Line[2] and laneA_Line[1] - 20 < cy < laneA_Line[1] + 20:
            if laneAcount.count(id) == 0:
                laneAcount.append(id)

        if laneB_Line[0] < cx < laneB_Line[2] and laneB_Line[1] - 20 < cy < laneB_Line[1] + 20:
            if laneBcount.count(id) == 0:
                laneBcount.append(id)

        

        cv2.circle(frm,(990,90),15,(66, 84, 245),-1)
        cv2.circle(frm,(990,130),15,(84, 245, 66),-1)
        cvzone.putTextRect(frm, f'LANE A Vehicles ={len(laneAcount)}', [1020, 99], thickness=4, scale=2.3, border=2)
        cvzone.putTextRect(frm, f'LANE B Vehicles ={len(laneBcount)}', [1020, 140], thickness=4, scale=2.3, border=2)

    cv2.imshow('frame', frm)
    cv2.waitKey(1)

vid.release()
cv2.destroyAllWindows()
