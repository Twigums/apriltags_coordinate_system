import cv2, apriltag, os
import numpy as np

from apriltag_functions import apriltag_init, detect_apriltag, draw_apriltag_bbox
from opencv_functions import get_camera_calibration, initialize_detector, get_detection, draw_polygon
from utility_functions import color_to_rgb, calculate_matrix_angle


USE_APRILTAGS = False
PATH_TO_CAMERA_CALIBRATION_FILE = "./calibration/20250408-Arducam.npz"
PATH_TO_TEMPLATES = "./templates"

cam = cv2.VideoCapture(0)

camera_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
camera_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

camera_matrix, new_camera_matrix, distortion_coefficients = get_camera_calibration(PATH_TO_CAMERA_CALIBRATION_FILE)
red_rgb = color_to_rgb("red")

sift, flann = initialize_detector()

# load all template images as numpy arrays
templates = []
template_filenames = [f for f in os.listdir(PATH_TO_TEMPLATES) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
for template_filename in template_filenames:
    path_to_template = os.path.join(PATH_TO_TEMPLATES, template_filename)
    image = cv2.imread(path_to_template)

    if image is not None:
        templates.append(image)

    else:
        print(f"Failed to load template at: {path_to_template}.")

if USE_APRILTAGS:
    at_detector = apriltag_init()
    tag_size = 0.02 # in meters

while True:
    ret, frame = cam.read()

    if not ret:
        break

    if USE_APRILTAGS:

        # apriltags needs input to be numpy.int8 grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        np_gray_image = np.asarray(gray_image)
    
        res = detect_apriltag(at_detector, np_gray_image)
    
        for tag in res:
            draw_apriltag_bbox(frame, tag, red_rgb)
            draw_axis(frame, tag, red_rgb)
            draw_depth(frame, tag, tag_size, camera_matrix, distorion_coefficients, red_rgb)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("t"):

        # if we have multiple templates, probably zip it with different colors?
        for template in templates:
            results, metrics = get_detection(sift, flann, template, frame)
    
            # its slower than [-1], but only works if we found a good iou early
            idx_best = np.argmax(metrics["iou"])
            best_H = results["H"][idx_best]
            best_transformed_box = results["transformed_box"][idx_best]
            best_iou = metrics["iou"][idx_best]
    
            draw_polygon(frame, best_transformed_box)
            theta = calculate_matrix_angle(best_H)
    
            print(f"IOU: {best_iou}, Angle: {theta}")

    cv2.imshow("camera", frame)

cam.release()
cv2.destroyAllWindows()