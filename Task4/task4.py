import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("yolov8l.pt")

input_dir = '/Users/dariadragomir/AI_siemens/Task4/input'
output_dir = '/Users/dariadragomir/AI_siemens/Task4/output'
os.makedirs(output_dir, exist_ok=True)

worse_polygon = np.array([(248, 219), (397, 228), (1252, 1044), (583, 1037)])
bad_polygon = np.array([(134, 10), (237, 7), (682, 219), (273, 208)])
park_polygon = np.array([(533, 223), (684, 226), (1908, 988), (1534, 1040)])
roi_polygon = np.array([(407, 229), (519, 234), (1526, 1043), (1261, 1029)])

dic = {}  # results 
bad_dic = {}  # eliminate 

def process_video(video_id):
    input_path = os.path.join(input_dir, f'{video_id:02d}.mp4')
    output_path = os.path.join(output_dir, f'output_{video_id:02d}.txt')

    cap = cv2.VideoCapture(input_path)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_cars = defaultdict(lambda: defaultdict(list))

    frame_idx = 0
    count_cars = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        results = model.track(frame, persist=True)
    
        annotated_frame = frame.copy()
        
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy.int().tolist()[0]
                class_id = int(box.cls[0])
                car_id = int(box.id.item())
                
                if class_id in [2, 7]:  # car or truck
                    all_cars[frame_idx][car_id] = xyxy

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
        if frame_idx > number_of_frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    car_center = [0, 0]
    
    for frame in all_cars:
        for c_id in all_cars[frame]:
            bbox = all_cars[frame][c_id]
            car_center[0] = (bbox[0] + bbox[2]) / 2
            car_center[1] = (bbox[1] + bbox[3]) / 2
        
            if cv2.pointPolygonTest(roi_polygon, (car_center[0], car_center[1]), False) >= 0:
                dic[c_id] = 1
            
            elif cv2.pointPolygonTest(park_polygon, (car_center[0], car_center[1]), False) >= 0 and c_id in dic:
                dic[c_id] = 0
            
            if cv2.pointPolygonTest(bad_polygon, (car_center[0], car_center[1]), False) >= 0:
                bad_dic[c_id] = -1
        
            if cv2.pointPolygonTest(worse_polygon, (car_center[0], car_center[1]), False) >= 0:
                bad_dic[c_id] = -1
            
    for car_id in dic:
        if dic[car_id] == 1 and bad_dic.get(car_id) != -1:
            count_cars += 1

    with open(output_path, 'w') as f:
        f.write(str(count_cars))
                
for video_id in range(1, 16):
    process_video(video_id)
