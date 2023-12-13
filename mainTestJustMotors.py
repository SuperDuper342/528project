import math
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import sleep
from adafruit_motorkit import MotorKit

def setup():
    # Setup Model

    # Setup MotorKit
    global kit
    kit = MotorKit()

    # Setup MiDaS
    global midas
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    midas.to('cpu')
    midas.eval()

    global transforms, transform
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transforms.small_transform

    global depth_to_distance_factor
    depth_to_distance_factor = 1.0 / 250.0

def reshapeForModel(frame):
    reshaped_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    reshaped_frame = cv2.resize(reshaped_frame, (224, 224))
    reshaped_frame = reshaped_frame / 255.0
    reshaped_frame = reshaped_frame.astype(np.float32)
    return reshaped_frame

def bboxCenterPoint(bbox):
    bbox_center_x = int((bbox[0] + bbox[2]) / 2)
    bbox_center_y = int((bbox[1] + bbox[3]) / 2)

    return [bbox_center_x, bbox_center_y]

def calculate_direction(X, frame_width=680):
    frame_center_x = frame_width / 2

    return X - frame_center_x

def determineDepth(frame, point):
    # Transform input for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()
    
    # Get the Depth
    depth = output[point[1], point[0]]

    # Converting the Depth
    distance = 1 / (depth * depth_to_distance_factor)

    # Display the result using Matplotlib
    plt.imshow(output)
    plt.scatter(*reversed(point), c='red', marker='x', label='Center of Object')
    plt.legend()
    plt.pause(0.00001)

    # Print the distance at the specified point
    print(f"Distance at point {point}: {distance:.2f} meters")

    return distance

def adjust_motors(x_component, depth, frame_width=224):
    if depth > 1.00:                        # Object is far, move towards it
        if x_component > 0:                 # Object is on the right of the center, turn right
            kit.motor1.throttle = None
            kit.motor2.throttle = -1     	# Motors 1 and 4 are placed in the front
            kit.motor3.throttle = -1      	# Motors 2 and 3 are placed in the opposite direction
            kit.motor4.throttle = 1
        elif x_component < 0:               # Object is on the left of the center, turn left
            kit.motor1.throttle = 1
            kit.motor2.throttle = -1
            kit.motor3.throttle = -1
            kit.motor4.throttle = None
        elif x_component == 0:              # Object is on the center axis, move forward
            kit.motor1.throttle = 1
            kit.motor2.throttle = -1
            kit.motor3.throttle = -1
            kit.motor4.throttle = 1

        sleep(1)

        kit.motor1.throttle = 0
        kit.motor2.throttle = 0
        kit.motor3.throttle = 0
        kit.motor4.throttle = 0

    elif depth <= 1.00:                     # Object is near, stop        
        kit.motor1.throttle = 0
        kit.motor2.throttle = 0
        kit.motor3.throttle = 0
        kit.motor4.throttle = 0

def append_coords_to_img(conf, coords):
    if conf < 0.9:
        return None
    
    x0 = int((coords[0][0])*680)
    y0 = int((coords[0][1])*440)
    x1 = int((coords[0][2])*680)
    y1 = int((coords[0][3])*440)
    
    print(x0, y0, x1, y1)

    return [x0, y0, x1, y1]

def main():
    # Run the setup for model, MotorKit, and MiDaS
    setup()

    arr = [[0.99973965, [0.09566244, 0.09882214, 0.43835637, 0.6129583]],   # Turn Left
           [0.99979246, [0.66429305, 0.22275576, 0.933295, 0.58063793]],    # Turn Right
           [0.9998633, [0.3553293, 0.53914285, 0.646554, 0.8747066]],       # Forward
           [0.00922467, [0.06606074, 0.08888596, 0.05474899, 0.11458659]]   # No Movement
           ]

    i = 0

    # Hook into openCV
    camStream = cv2.VideoCapture(0)

    while camStream.isOpened():
        ret, frame = camStream.read()

        if i is 4:
            i = 0
        else:
            i = i + 1

        # Calculate the bbox
        confidence = arr[i][0]
        coords = arr[i][0]
        
        bbox = append_coords_to_img(confidence, coords)

        # Determine direction of turning
        if bbox is not None:
            bboxCenter = bboxCenterPoint(bbox)

            # Determine direction of turning
            vector_x = calculate_direction(bboxCenter[0])
            
            # Determine depth
            depth = determineDepth(frame, bboxCenter)

            # Adjust the motors
            adjust_motors(vector_x, depth)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    # Release resources
    camStream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
