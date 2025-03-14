import cv2, apriltag
import numpy as np

from apriltag_functions import *

PATH_TO_CAMERA_CALIBRATION_FILE = "./calibration/20250310-Arducam.npz"
cam = cv2.VideoCapture(0)

camera_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

camera_M, distort_coeff = get_camera_calibration(PATH_TO_CAMERA_CALIBRATION_FILE)
at_detector = apriltag_init()

red_rgb = color_to_rgb("red")
tag_size = 0.02 # in meters
text_offset = 10
arrow_scale = 0.1

while True:
    ret, frame = cam.read()

    if not ret:
        break

    # apriltags needs input to be numpy.int8 grayscale
    np_gray_frame = img_to_np_gray(frame)

    res = detect_apriltag(at_detector, np_gray_frame)

    for tag in res:
        draw_bbox(frame, tag, red_rgb)
        draw_axis(frame, tag, arrow_scale, text_offset, red_rgb)
        draw_depth(frame, tag, tag_size, camera_M, distort_coeff, red_rgb)

    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
