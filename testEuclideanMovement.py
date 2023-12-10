import math
import time
from adafruit_motorkit import MotorKit

# Create a motor kit object
kit = MotorKit()

def calculate_direction(bbox, frame_width, frame_height):
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2

    bbox_center_x *= frame_width
    bbox_center_y *= frame_width

    frame_center_x = 112
    frame_center_y = 112

    # Calculate vector from frame center to object center
    vector_x = bbox_center_x - frame_center_x
    vector_y = bbox_center_y - frame_center_y

    # Only return the x component
    return vector_x

def adjust_motors(x_component, frame_width):
    max_x_component = frame_width / 2
    max_pwm = 1.0
    min_pwm = 0.3

    pwm = (x_component / max_x_component) * (max_pwm - min_pwm) + min_pwm

    # Adjust motors based on x component
    if x_component > 0:  # Object is on the right
        # Turn left
        kit.motor1.throttle = pwm
        kit.motor2.throttle = -pwm  # Adjusted for opposite direction
        kit.motor3.throttle = pwm   # Adjusted for opposite direction
        kit.motor4.throttle = -pwm
    else:
        # Turn right
        kit.motor1.throttle = -pwm
        kit.motor2.throttle = pwm   # Adjusted for opposite direction
        kit.motor3.throttle = -pwm  # Adjusted for opposite direction
        kit.motor4.throttle = pwm

    # Calculate duration based on the distance from the center
    distance_from_center = abs(x_component)
    max_distance = frame_width / 2
    max_duration = 1.0  # Adjust based on your desired maximum duration

    duration = (distance_from_center / max_distance) * max_duration

    # Sleep for the calculated duration
    time.sleep(duration)

    # Stop the motors
    kit.motor1.throttle = 0
    kit.motor2.throttle = 0
    kit.motor3.throttle = 0
    kit.motor4.throttle = 0

if __name__ == "__main__":
    # Replace this with the actual bounding box coordinates
    example_bbox1 = [0.0210451, 0.07524262, 0.18377778, 0.35889962] #located on the left side, should turn right
    example_bbox2 = [0.5456728, 0.19789025, 0.8760194, 0.6572715] #located on the right side, should turn left

    # Replace this with the actual frame width and height
    frame_width = 224  # Example frame width
    frame_height = 224  # Example frame height

    # Get x component with scaled bounding box coordinates
    x_component = calculate_direction(example_bbox1, frame_width, frame_height)

    adjust_motors(x_component, frame_width)
