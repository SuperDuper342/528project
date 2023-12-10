import math
import time
from adafruit_motorkit import MotorKit

# Create a motor kit object
kit = MotorKit()

def calculate_direction(bbox, frame_width, frame_height):
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2

    # No need to scale the coordinates since the frame size is 224x224

    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

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
        # Turn right
        kit.motor1.throttle = pwm
        kit.motor2.throttle = -pwm
        kit.motor3.throttle = -pwm
        kit.motor4.throttle = pwm
    else:
        # Turn left
        kit.motor1.throttle = -pwm
        kit.motor2.throttle = pwm
        kit.motor3.throttle = pwm
        kit.motor4.throttle = -pwm

if __name__ == "__main__":
    # Replace this with the actual bounding box coordinates
    example_bbox = [0.0210451, 0.07524262, 0.18377778, 0.35889962]  # Example scaled coordinates

    # Replace this with the actual frame width and height
    frame_width = 224  # Example frame width
    frame_height = 224  # Example frame height

    # Get x component with scaled bounding box coordinates
    x_component = calculate_direction(example_bbox, frame_width, frame_height)

    adjust_motors(x_component, frame_width)
