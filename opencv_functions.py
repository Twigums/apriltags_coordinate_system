import cv2
import numpy as np


def calibrate_camera(path_to_checkerboard_image, checkerboard_size, path_to_save_npz):
    checkerboard_units_x, checkerboard_units_y = checkerboard_size
    checkerboard_img = cv2.imread(path_to_checkerboard_image)
    height, width = checkerboard_img.shape[:2]

    gray_image = cv2.cvtColor(checkerboard_img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray_image, (checkerboard_units_x, checkerboard_units_y), None)

    if found == False:
        raise ValueError("Found != True -> Checkerboard was not found.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((checkerboard_units_y * checkerboard_units_x, 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_units_x, 0:checkerboard_units_y].T.reshape(-1, 2)
    
    objpoints = []
    imgpoints = []
    
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1,-1), criteria)
    imgpoints.append(corners2)

    # we can finally calibrate
    res, camera_M, distort_coeff, rot_v, trans_v = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)
    new_camera_M, roi = cv2.getOptimalNewCameraMatrix(camera_M, distort_coeff, (width, height), 1, (width, height))

    np.savez(path_to_save_npz, camera_M = camera_M, new_camera_M = new_camera_M, distort_coeff = distort_coeff)

    return True

def get_camera_calibration(filepath):
    npzfile = np.load(filepath)
    camera_matrix = npzfile["camera_M"]
    new_camera_matrix = npzfile["new_camera_M"]
    distortion_coefficients = npzfile["distort_coeff"]

    return camera_matrix, new_camera_matrix, distortion_coefficients

def img_to_np_gray(img: np.ndarray) -> np.ndarray:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    np_gray_img = np.asarray(gray_img)

    return np_gray_img

def draw_bbox(img: np.ndarray, tag: apriltag.Detection, color: tuple[int, int, int]) -> None:
    corner_points = tag.corners.astype(int)
    center = tag.center.astype(int)

    cv2.polylines(img, [corner_points], isClosed = True, color = color, thickness = 1)
    cv2.circle(img, center, 1, color, -1)

def draw_axis(img: np.ndarray, tag: apriltag.Detection, arrow_scale: float, text_offset: int, color: tuple[int, int, int]) -> None:
    corner_points = tag.corners.astype(int)
    center = tag.center.astype(int)
    x_vec = corner_points[1] - corner_points[0]
    y_vec = corner_points[1] - corner_points[2]

    x_unit = x_vec / np.linalg.norm(x_vec)
    y_unit = y_vec / np.linalg.norm(y_vec)

    arrow_length = int(min(np.linalg.norm(x_vec), np.linalg.norm(y_vec)) * arrow_scale)

    x_label_pos = center + x_unit * arrow_length + np.array([text_offset, 0])
    y_label_pos = center + y_unit * arrow_length + np.array([0, text_offset])

    cv2.arrowedLine(img, tuple(center), tuple((center + x_unit * arrow_length).astype(int)), color, 2, tipLength = 0.1)
    cv2.arrowedLine(img, tuple(center), tuple((center + y_unit * arrow_length).astype(int)), color, 2, tipLength = 0.1)

    cv2.putText(img, "x", tuple(x_label_pos.astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
    cv2.putText(img, "y", tuple(y_label_pos.astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

def draw_depth(img: np.ndarray, tag: apriltag.Detection, tag_size: float, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray, color: tuple[int, int, int]) -> None:
    objPoints = np.array([
        [-tag_size / 2, -tag_size / 2, 0],
        [ tag_size / 2, -tag_size / 2, 0],
        [ tag_size / 2,  tag_size / 2, 0],
        [-tag_size / 2,  tag_size / 2, 0]
    ])

    corner_points = tag.corners

    image_points = corner_points.reshape(-1, 2)
    res, rvec, tvec = cv2.solvePnP(objPoints, image_points, camera_matrix, distortion_coefficients)
    depth = tvec[2][0]

    if res and depth > 0:
        cv2.putText(img, f"Depth: {depth:.2f}m", tuple(corner_points[2].astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)