import math
import time
from adafruit_motorkit import MotorKit

# Create a motor kit object
kit = MotorKit()

def calculate_distance(bbox, frame_width, frame_height):
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2

    # Scale the coordinates by the frame width
    bbox_center_x *= frame_width
    bbox_center_y *= frame_width

    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    distance = math.sqrt((bbox_center_x - frame_center_x)**2 + (bbox_center_y - frame_center_y)**2)

    return distance

def adjust_motors(distance, frame_width):
    max_distance = frame_width / 2
    max_pwm = 1.0
    min_pwm = 0.3

    pwm = (distance / max_distance) * (max_pwm - min_pwm) + min_pwm

    # If the object is along the center axis, move forward until a certain distance
    if pwm > 0.8:  # Adjust this threshold based on your requirements
        kit.motor1.throttle = pwm
        kit.motor2.throttle = pwm
        kit.motor3.throttle = pwm
        kit.motor4.throttle = pwm
        time.sleep(1)  # Adjust the time based on how long you want to move forward
        kit.motor1.throttle = 0
        kit.motor2.throttle = 0
        kit.motor3.throttle = 0
        kit.motor4.throttle = 0
    else:
        # Adjust motors based on distance
        if distance > 50:  # Example threshold, adjust based on your requirements
            kit.motor1.throttle = pwm
            kit.motor2.throttle = -pwm
            kit.motor3.throttle = -pwm
            kit.motor4.throttle = pwm
        else:
            kit.motor1.throttle = -pwm
            kit.motor2.throttle = pwm
            kit.motor3.throttle = pwm
            kit.motor4.throttle = -pwm

if __name__ == "__main__":
    # Replace this with the actual bounding box coordinates
    example_bbox = [0.0210451, 0.07524262, 0.18377778, 0.35889962]

    # Replace this with the actual frame width and height
    frame_width = 224  # Example frame width
    frame_height = 224  # Example frame height

    # Get distance with scaled bounding box coordinates
    distance = calculate_distance(example_bbox, frame_width, frame_height)

    adjust_motors(distance, frame_width)
