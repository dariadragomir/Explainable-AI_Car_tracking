import os
import cv2
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("yolov8l.pt")

input_dir = '/Users/dariadragomir/AI_siemens/Task3/test'
output_dir = '/Users/dariadragomir/AI_siemens/Task3/output_test'

os.makedirs(output_dir, exist_ok=True)

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def process_video(video_id):
    video_path = os.path.join(input_dir, f'{video_id:02d}.mp4')
    txt_path = os.path.join(input_dir, f'{video_id:02d}.txt')
    output_path = os.path.join(output_dir, f'output_{video_id:02d}.txt')

    with open(txt_path, 'r') as file:
        lines = file.readlines()
    number_of_frames = int(lines[0].split()[0])
    initial_bbox = list(map(int, lines[1].split()[1:]))

    cap = cv2.VideoCapture(video_path)
    all_cars = defaultdict(lambda: defaultdict(list))

    frame_idx = 0
    
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
                    
                   cv2.rectangle(annotated_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                   cv2.putText(annotated_frame, f'ID: {car_id}', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
      
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
        
        frame_idx += 1
        if frame_idx > number_of_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Identify the car closest to the initial bounding box
    found_car_id = 0
    best_iou = 0
    
    for frame in all_cars:
        if frame >100:
            break
        for c_id in all_cars[frame]:
            bbox = all_cars[frame][c_id]
            iou = bb_intersection_over_union(bbox, initial_bbox)
            if iou > best_iou:
                best_iou = iou
                found_car_id = c_id
    ok_print = 1
    
    with open(output_path, 'w') as f:
        f.write(f'{number_of_frames} -1 -1 -1 -1\n')
        for frame_idx in range(number_of_frames):
            bbox = all_cars[frame_idx][found_car_id]
            if bbox:
                f.write(f'{frame_idx} {" ".join(map(str, bbox))}\n')
                ok_print = 0
            elif ok_print == 1: #bbox is empty and i didnt print anything = car not found yet
                f.write(f'{frame_idx} {" ".join(map(str, initial_bbox))}\n')
                
for video_id in range(1, 16):
    process_video(video_id)
