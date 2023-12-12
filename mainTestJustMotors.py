import time
import math
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
    # Resize the frame to (224, 224)
    resized_frame = cv2.resize(frame, (224, 224))

    # Expand dimensions to match the target shape (1, 224, 224, 3)
    reshaped_frame = np.expand_dims(resized_frame, axis=0)

    # Optionally, you can normalize the pixel values to be in the range [0, 1]
    reshaped_frame = reshaped_frame / 255.0

    return reshaped_frame

def bboxCenterPoint(bbox, conf):
    if conf < 0.90:
        return False, None
    else:
        bbox_center_x = int(((bbox[0] + bbox[2]) / 2) * 224)
        bbox_center_y = int(((bbox[1] + bbox[3]) / 2) * 224)

        return True, [bbox_center_x, bbox_center_y]

def calculate_direction(X, frame_width=224):
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

        time.sleep(1)

        kit.motor1.throttle = 0
        kit.motor2.throttle = 0
        kit.motor3.throttle = 0
        kit.motor4.throttle = 0

    elif depth <= 1.00:                     # Object is near, stop        
        kit.motor1.throttle = 0
        kit.motor2.throttle = 0
        kit.motor3.throttle = 0
        kit.motor4.throttle = 0

def main():
    # Run the setup for model, MotorKit, and MiDaS
    setup()

    # Hook into openCV
    camStream = cv2.VideoCapture(0)

    while camStream.isOpened():
        ret, frame = camStream.read()

        # reshape frame for model
        modelFrame = reshapeForModel(frame)

        # TestValues
        bbox1 = [0.09566244, 0.09882214, 0.43835637, 0.6129583]
        bbox2 = [0.66429305, 0.22275576, 0.933295, 0.58063793]
        bbox3 = [0.3553293, 0.53914285, 0.646554, 0.8747066]
        bbox4 = [0.06606074, 0.08888596, 0.05474899, 0.11458659]

        conf1 = 0.99973965
        conf2 = 0.99979246
        conf3 = 0.9998633
        conf4 = 0.00922467

        # Calculate the centerpoints of the bbox
        bool1, bboxCenter1 = bboxCenterPoint(bbox1, conf1)
        bool2, bboxCenter2 = bboxCenterPoint(bbox2, conf2)
        bool3, bboxCenter3 = bboxCenterPoint(bbox3, conf3)
        bool4, bboxCenter4 = bboxCenterPoint(bbox4, conf4)

        # Determine direction of turning
        if bool1 is True:
            x1 = calculate_direction(bboxCenter1[0])
            depth1 = determineDepth(modelFrame, bboxCenter1)
            adjust_motors(x1, depth1)   # Should turn left
            time(1)
        elif bool1 is not True:
            print("No Object Detected!")

        if bool2 is True:
            depth2 = determineDepth(modelFrame, bboxCenter2)
            x2 = calculate_direction(bboxCenter2[0])
            adjust_motors(x2, depth2)   # Should turn right
            time(1)
        elif bool2 is not True:
            print("No Object Detected!")

        if bool3 is True:
            x3 = calculate_direction(bboxCenter3[0])
            depth3 = determineDepth(modelFrame, bboxCenter3)
            adjust_motors(x3, depth3)   # Should go forward
            time(1)
        elif bool3 is not True:
            print("No Object Detected!")

        if bool4 is True:
            x4 = calculate_direction(bboxCenter4[0])
            depth4 = determineDepth(modelFrame, bboxCenter4)
            adjust_motors(x4, depth4)   # Should stop
            time(1)
        elif bool4 is not True:
            print("No Object Detected!")
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    # Release resources
    camStream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
