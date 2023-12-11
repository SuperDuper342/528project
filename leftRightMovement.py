# motor_control.py

from adafruit_motorkit import MotorKit

def calculate_direction(bbox, frame_width, frame_height):
    bbox_center_x = (bbox[0] + bbox[2]) / 2

    frame_center_x = frame_width / 2

    vector_x = bbox_center_x - frame_center_x

    return vector_x

def adjust_motors(x_component, frame_width):
    max_x_component = frame_width / 2
    max_pwm = 1.0
    min_pwm = 0.3

    pwm = (x_component / max_x_component) * (max_pwm - min_pwm) + min_pwm

    if x_component > 0:  # Object is on the right
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
    # Example usage if the script is executed directly
    kit = MotorKit()

    example_bbox = [0.0210451, 0.07524262, 0.18377778, 0.35889962]
    frame_width = 224
    frame_height = 224

    x_component = calculate_direction(example_bbox, frame_width, frame_height)
    adjust_motors(x_component, frame_width)
