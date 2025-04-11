import cv2, apriltag
import numpy as np

from apriltag_functions import *
from opencv_functions import *
from utility_functions import color_to_rgb


PATH_TO_CAMERA_CALIBRATION_FILE = "./calibration/20250408-Arducam.npz"
cam = cv2.VideoCapture(0)

camera_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

camera_matrix, new_camera_matrix, distortion_coefficients = get_camera_calibration(PATH_TO_CAMERA_CALIBRATION_FILE)
at_detector = apriltag_init()

red_rgb = color_to_rgb("red")
tag_size = 0.02 # in meters


while True:
    ret, frame = cam.read()

    if not ret:
        break

    # apriltags needs input to be numpy.int8 grayscale
    np_gray_frame = img_to_np_gray(frame)

    res = detect_apriltag(at_detector, np_gray_frame)

    for tag in res:
        draw_apriltag_bbox(frame, tag, red_rgb)
        draw_axis(frame, tag, red_rgb)
        draw_depth(frame, tag, tag_size, camera_matrix, distorion_coefficients, red_rgb)

    cv2.imshow("camera", frame) 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()