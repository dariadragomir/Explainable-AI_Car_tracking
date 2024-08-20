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
    if filename.endswith('.jpg'):
        image_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        query_file_path = os.path.join(input_dir, f"{base_name}_query.txt")
        output_file_path = os.path.join(output_dir, f"{base_name}_output.txt")

        if not os.path.exists(query_file_path):
            print(f"Query file {query_file_path} not found. Skipping {filename}.")
            continue

        image = cv2.imread(image_path)
        
        results = model(image)
        
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

                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f'{label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        for id_parking_spot, parking_spot in enumerate(parking_spots):
            spot_occupied = 0
            for car_center in cars_centers:
                if cv2.pointPolygonTest(parking_spot, (car_center[0], car_center[1]), False) >= 0:
                    spot_occupied = 1
                    break
            status_parking_spots[id_parking_spot] = spot_occupied
        
        with open(query_file_path, 'r') as query_file, open(output_file_path, 'w') as output_file:
            n = int(query_file.readline().strip())
            output_file.write(str(n) + '\n')
            for _ in range(n):
                spot_id = int(query_file.readline().strip())
                status = '1' if status_parking_spots[spot_id - 1] == 1 else '0'
                output_file.write(str(spot_id)+ ' ' + str(status) + '\n')

        output_image_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_image_path, image)

print("Detection complete. Processed images and output files are saved in the output directory.")
