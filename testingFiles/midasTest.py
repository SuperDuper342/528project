import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

# Download the MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Hook into OpenCV
cap = cv2.VideoCapture(0)


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Transform input for midas
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

            depth_value = prediction[y, x].item()
            print(f"Depth at pixel ({x}, {y}): {depth_value:.2f} meters")

            # Visualize depth map
            depth_colormap = cv2.applyColorMap(
                np.uint8(255 * (1 - prediction / prediction.max())),
                cv2.COLORMAP_JET
            )
            cv2.imshow('Depth Map', depth_colormap)


cv2.namedWindow('CV2Frame')
cv2.setMouseCallback('CV2Frame', on_mouse_click)

while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow('CV2Frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

cv2.destroyAllWindows()
