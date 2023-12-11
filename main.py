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
    kit = MotorKit

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

def bboxCenterPoint(bbox):
    bbox_center_x = ((bbox[0] + bbox[2]) / 2) * 224
    bbox_center_y = ((bbox[1] + bbox[3]) / 2) * 224

    return [bbox_center_x, bbox_center_y]

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
    max_x = frame_width/2
    max_pwm = 1.0
    min_pwm = 0.3

    pwm = (x_component / max_x) * (max_pwm - min_pwm) + min_pwm

    if depth > 1.00:                        # Object is far, move towards it
        if x_component > 0:                 # Object is on the right of the center, turn right
            kit.motor1.throttle = None
            kit.motor2.throttle = -pwm      # Motors 1 and 4 are placed in the front
            kit.motor3.throttle = -pwm      # Motors 2 and 3 are placed in the opposite direction
            kit.motor4.throttle = pwm
        elif x_component < 0:               # Object is on the left of the center, turn left
            kit.motor1.throttle = pwm
            kit.motor2.throttle = -pwm
            kit.motor3.throttle = -pwm
            kit.motor4.throttle = None
        elif x_component == 0:              # Object is on the center axis, move forward
            kit.motor1.throttle = pwm
            kit.motor2.throttle = -pwm
            kit.motor3.throttle = -pwm
            kit.motor4.throttle = pwm

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

        # Model goes here to check for object

        # TestValues
        # bbox = [0.0210451, 0.07524262, 0.18377778, 0.35889962]

        # Calculate the centerpoints of the bbox
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
