import os
import cv2
import numpy as np
from ultralytics import YOLO

parking_spots = [
    np.array([(1615, 755), (1758, 838), (1759, 1050), (1553, 1010)]),
    np.array([(1484, 714), (1603, 771), (1550, 1017), (1402, 918)]),
    np.array([(1360, 673), (1467, 733), (1393, 912), (1267, 822)]),
    np.array([(1258, 645), (1339, 687), (1258, 807), (1160, 745)]),
    np.array([(1173, 591), (1252, 636), (1160, 735), (1084, 678)]),
    np.array([(1094, 544), (1163, 586), (1094, 655), (1019, 606)]),
    np.array([(1022, 501), (1083, 537), (1030, 588), (952, 561)]),
    np.array([(958, 459), (1021, 495), (963, 546), (896, 530)]),
    np.array([(910, 421), (958, 452), (901, 511), (842, 483)]),
    np.array([(853, 387), (910, 415), (853, 470), (806, 454)])
]

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, 'input')
output_dir = os.path.join(script_dir, 'output')

model = YOLO('yolov9e.pt')

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.mp4'):
        print(f"Processing video: {filename}")
        video_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_file_path = os.path.join(output_dir, f"{base_name}_output2.txt")

        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Frame count for {filename}: {frame_count}")
        current_frame = 0

        while current_frame < frame_count:
            ret, frame = cap.read()
            current_frame += 1

            if not ret:
                break

            if current_frame < frame_count:
                continue
              
            print(f"Processing last frame of {filename}")
            results = model(frame)

            boxes = results[0].boxes.xyxy.numpy()
            scores = results[0].boxes.conf.numpy()
            class_ids = results[0].boxes.cls.numpy()

            cars_centers = []
            status_parking_spots = [0] * len(parking_spots)

            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = box
                label = model.names[int(class_id)]

                if label == 'car' or label == 'truck':  
                    car_center = [(x1 + x2) / 2, (y1 + y2) / 2]
                    cars_centers.append(car_center)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            for id_parking_spot, parking_spot in enumerate(parking_spots):
                spot_occupied = 0
                for car_center in cars_centers:
                    if cv2.pointPolygonTest(parking_spot, (car_center[0], car_center[1]), False) >= 0:
                        spot_occupied = 1
                        break
                status_parking_spots[id_parking_spot] = spot_occupied

            with open(output_file_path, 'w') as output_file:
                output_file.write('\n'.join(map(str, status_parking_spots)))

            output_image_path = os.path.join(output_dir, f"{base_name}_last_frame.jpg")
            print(f"Saving output image to: {output_image_path}")
            cv2.imwrite(output_image_path, frame)

        cap.release()

print("Detection complete. Processed videos and output files are saved in the output directory.", end='')
