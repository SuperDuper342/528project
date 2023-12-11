# midas_processor.py
import cv2
import torch
import matplotlib.pyplot as plt

def process_video(point_coordinates):
    # Download the MiDaS
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    midas.to('cpu')
    midas.eval()

    # Input transformation pipeline
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transforms.small_transform

    # Hook into OpenCV
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return

    # Conversion factor for depth values to represent distance in meters
    depth_to_distance_factor = 1.0 / 250  # You may need to adjust this based on your specific scenario

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break

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

        # Get depth at the specified point
        depth_at_point = output[point_coordinates[1], point_coordinates[0]]

        # Convert depth to distance in meters
        distance_at_point = 1 / (depth_at_point * depth_to_distance_factor)

        # Display the result using Matplotlib
        plt.imshow(output)
        plt.scatter(*reversed(point_coordinates), c='red', marker='x', label='Point of Interest')
        plt.legend()
        plt.pause(0.00001)

        # Print the distance at the specified point
        print(f"Distance at point {point_coordinates}: {distance_at_point:.2f} meters")

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example point coordinates
    example_point_coordinates = (100, 150)
    
    # Run MiDA processing on the video stream
    process_video(example_point_coordinates)
