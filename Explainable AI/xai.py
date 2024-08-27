from ultralytics import YOLO
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
from lime import lime_image
clicked_points = [] 
pause = False
ok = 0

def xai(frame, bbox):
    img = frame
    bbox = bbox.xyxy.tolist()[0]
    for i in range(4):
        bbox[i] = int(bbox[i])
    print(bbox)
    img = img[bbox[1]-100:bbox[3]+100, bbox[0]-100:bbox[2]+100]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img) # display the image
    print(img)
    img.show()
    
    # resize and take the center part of image to what our model expects
    def get_input_transform():
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])       
        transf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])    

        return transf

    def get_input_tensors(img):
        transf = get_input_transform()
        # unsqeeze converts single image to batch of 1
        return transf(img).unsqueeze(0)

    model = models.inception_v3(pretrained=True)

    idx2label, cls2label, cls2idx = [], {}, {}
    with open(os.path.abspath('./data/imagenet_class_index.json'), 'r') as read_file:
        class_idx = json.load(read_file)
        idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
        cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}    
        
    img_t = get_input_tensors(img)
    model.eval()
    logits = model(img_t)

    probs = F.softmax(logits, dim=1)
    probs5 = probs.topk(5)
    tuple((p,c, idx2label[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy()))

    def get_pil_transform(): 
        transf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224)
        ])    

        return transf

    def get_preprocess_transform():
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])     
        transf = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])    

        return transf    

    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()

    def batch_predict(images):
        model.eval()
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch = batch.to(device)
        
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    test_pred = batch_predict([pill_transf(img)])
    test_pred.squeeze().argmax()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(img)), 
                                            batch_predict, # classification function
                                            top_labels=5, 
                                            hide_color=0, 
                                            num_samples=1000) # number of images that will be sent to classification function

    from skimage.segmentation import mark_boundaries
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    plt.imshow(img_boundry1)
    plt.show()

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=15, hide_rest=False)
    img_boundry2 = mark_boundaries(temp/255.0, mask)
    plt.imshow(img_boundry2)
    plt.show()
        
    
def is_point_inside_bbox(px, py, bbox):     
    x1, y1, x2, y2 = bbox    
    return x1 <= px <= x2 and y1 <= py <= y2

def functie(frame, point):
    model = YOLO("yolov8n.pt")
    results = model(frame)  
    boxes = results[0].boxes
    for bbox in boxes:
        if is_point_inside_bbox(point[0], point[1], bbox.xyxy.cpu()[0]):
            xai(frame, bbox)
            exit()
        #result.show() 
        
def click_event(event, x, y, flags, param):     
    global pause
    if event == cv2.EVENT_LBUTTONDOWN:         # Store the coordinates of the click        
        clicked_points.append((x, y))        
        #print(f"Clicked at: ({x}, {y})") # Load the video
        pause = True
        
video = cv2.VideoCapture('data/01.mp4')  # Replace with your video file path
# Set the callback function to the window
cv2.namedWindow("Video") 
cv2.setMouseCallback("Video", click_event) 
while video.isOpened():   
    if not pause:
        ret, frame = video.read()
        if not ret:
            break  
        cv2.imshow("Video", frame)     # Wait for 25 ms before moving on to the next frame, exit if 'q' is pressed
    
    key = cv2.waitKey(25) & 0XFF
    if clicked_points and pause and ok == 0:
        ok = 1
        point = clicked_points[-1]
        
        functie(frame, point)
        print(clicked_points[-1])
        
    if key == ord('q'):
        break # Quit if 'q' is pressed 
    elif key == ord('p'):
        pause = not pause # Toggle pause if 'p' is pressed
        ok = 0
    
video.release() 
cv2.destroyAllWindows() # Print all recorded click points
print("All clicked points:", clicked_points)
