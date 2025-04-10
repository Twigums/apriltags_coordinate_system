# istg training a model wouldve been easier than doing this lol

import apriltag, cv2
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

def undistort_image(image, camera_matrix, distortion_coefficients):
    height, width = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (width, height), 1, (width, height))
    x, y, roi_width, roi_height = roi
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, new_camera_matrix)
    undistorted_image = undistorted_image[y:y + roi_height, x:x + roi_width]

    return undistorted_image

def initialize_detector(flann_index_kdtree: int = 1, flann_trees: int = 5, flann_checks: int = 50):
    sift = cv2.SIFT_create()
    
    index_params = dict(algorithm = flann_index_kdtree, trees = flann_trees)
    search_params = dict(checks = flann_checks)
     
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    return sift, flann
    
# https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
def get_detection(sift, flann, template, target, iterations: int = 10, flann_k: int = 2, min_match_count: int = 10, matching_ratio_score: float = 0.7, ransac_threshold: float = 5.0):
    
    # find the keypoints and descriptors with SIFT
    kp_template, des_template = sift.detectAndCompute(template, None)
    kp_target, des_target = sift.detectAndCompute(target, None)
    
    all_H = []

    for i in range(iterations): 
        matches = flann.knnMatch(des_template, des_target, k = flann_k)

        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for m, n in matches:
            if m.distance < matching_ratio_score * n.distance:
                good_matches.append(m)

        if len(good_matches) > min_match_count:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
         
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
            all_H.append(H)
         
        else:
            print(f"Not enough matches are found in iter {i}: {len(good_matches)}/{min_match_count}")

    h, w, _ = template.shape # not greyscale, so 3 channels
    box_points = np.float32([
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0]
        ]).reshape(-1, 1, 2)

    avg_H = np.mean(np.array(all_H), axis = 0)
    transformed_box = cv2.perspectiveTransform(box_points, avg_H)
    
    return transformed_box

def img_to_np_gray(img: np.ndarray) -> np.ndarray:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    np_gray_img = np.asarray(gray_img)

    return np_gray_img

def draw_polygon(img: np.ndarray, points: np.ndarray, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 3) -> None:
    cv2.polylines(img, [np.int32(points)], True, color = color, thickness = thickness)

def draw_apriltag_bbox(img: np.ndarray, tag: apriltag.Detection, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 1) -> None:
    corner_points = tag.corners.astype(int)
    center = tag.center.astype(int)

    cv2.polylines(img, [corner_points], isClosed = True, color = color, thickness = thickness)
    cv2.circle(img, center, 1, color, -1)

def draw_axis(img: np.ndarray, tag: apriltag.Detection, color: tuple[int, int, int] = (0, 255, 0), arrow_scale: float = 0.1, text_offset: int = 10) -> None:
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

def draw_depth(img: np.ndarray, tag: apriltag.Detection, tag_size: float, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray, color: tuple[int, int, int] = (0, 255, 0)) -> None:
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