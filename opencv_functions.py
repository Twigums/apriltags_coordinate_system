# istg training a model wouldve been easier than doing this lol

import apriltag, cv2
import numpy as np


# finds checkerboard image and uses cv2 to find camera calibration matrices and distortion coefficients
# saves result as a .npz file
def calibrate_camera(path_to_checkerboard_image: str,
                     checkerboard_size: tuple[int, int],
                     path_to_save_npz: str) -> None:
    
    checkerboard_image = cv2.imread(path_to_checkerboard_image)
    height, width = checkerboard_image.shape[:2]

    gray_image = cv2.cvtColor(checkerboard_image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray_image, checkerboard_size, None)

    if found == False:
        raise ValueError("Found != True -> Checkerboard was not found.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((checkerboard_units_y * checkerboard_units_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_units_x, 0:checkerboard_units_y].T.reshape(-1, 2)
    
    objpoints = []
    imagepoints = []
    
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
    imagepoints.append(corners2)

    # we can finally calibrate
    res, camera_M, distort_coeff, rot_v, trans_v = cv2.calibrateCamera(objpoints, imagepoints, gray_image.shape[::-1], None, None)
    new_camera_M, roi = cv2.getOptimalNewCameraMatrix(camera_M, distort_coeff, (width, height), 1, (width, height))

    np.savez(path_to_save_npz, camera_M = camera_M, new_camera_M = new_camera_M, distort_coeff = distort_coeff)

# loads .npz file and returns the 3 saved numpy arrays
def get_camera_calibration(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    npzfile = np.load(filepath)
    camera_matrix = npzfile["camera_M"]
    new_camera_matrix = npzfile["new_camera_M"]
    distortion_coefficients = npzfile["distort_coeff"]

    return camera_matrix, new_camera_matrix, distortion_coefficients

# undistort an image by using camera numpy arrays
# returns the undistorted image as a numpy array
def undistort_image(image: np.ndarray, 
                    camera_matrix: np.ndarray, 
                    distortion_coefficients: np.ndarray) -> np.ndarray:
    
    height, width = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (width, height), 1, (width, height))
    x, y, roi_width, roi_height = roi
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, new_camera_matrix)
    undistorted_image = undistorted_image[y:y + roi_height, x:x + roi_width]

    return undistorted_image

# rotates image :)
def rotate_image(image: np.ndarray,
                 angle: float) -> np.ndarray:
    
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated_image

# easily initialize sift + flann
def initialize_detector(flann_index_kdtree: int = 1,
                        flann_trees: int = 5,
                        flann_checks: int = 50):

    sift = cv2.SIFT_create()
    
    index_params = dict(algorithm = flann_index_kdtree, trees = flann_trees)
    search_params = dict(checks = flann_checks)
     
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    return sift, flann

# uses sift and flann to match keypoints between template and target
# repeats "iterations" times and returns a transformed ROI in the detected object's coords
# https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
def get_detection(sift: cv2.SIFT,
                  flann: cv2.FlannBasedMatcher,
                  template: np.ndarray, 
                  target: np.ndarray,
                  iterations: int = 10,
                  flann_k: int = 2,
                  min_match_count: int = 10,
                  matching_ratio_score: float = 0.7,
                  ransac_threshold: float = 5.0):
    
    # find the keypoints and descriptors with SIFT
    kp_template, des_template = sift.detectAndCompute(template, None)
    kp_target, des_target = sift.detectAndCompute(target, None)
    
    all_H = []

    # repeating a few times should make the process more repeatable
    for i in range(iterations): 
        matches = flann.knnMatch(des_template, des_target, k = flann_k)

        # store all the good matches as per Lowe's ratio test.
        good_matches = []

        # only consider "good matches"
        for m, n in matches:
            if m.distance < matching_ratio_score * n.distance:
                good_matches.append(m)

        # find the transformation matrix, H
        if len(good_matches) > min_match_count:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
         
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
            all_H.append(H)
         
        else:
            print(f"Not enough matches are found in iter {i}: {len(good_matches)}/{min_match_count}")

    # i found that this code doesn't work for arbitrary x, y values
    # it seems much better if origin at (0, 0)
    h, w, _ = template.shape # not greyscale, so 3 channels
    box_points = np.float32([
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0]
        ]).reshape(-1, 1, 2)

    # apply transformation to rectangle
    avg_H = np.mean(np.array(all_H), axis = 0)
    transformed_box = cv2.perspectiveTransform(box_points, avg_H)
    
    return transformed_box

# converts image to greyscale image
def image_to_np_gray(image: np.ndarray) -> np.ndarray:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    np_gray_image = np.asarray(gray_image)

    return np_gray_image

# draws lines connecting to points on image
def draw_polygon(image: np.ndarray,
                 points: np.ndarray,
                 color: tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 3) -> None:

    cv2.polylines(image, [np.int32(points)], True, color = color, thickness = thickness)

# draw a box around detected apriltag
# not really a bbox because it rotates, but who cares :D
def draw_apriltag_bbox(image: np.ndarray,
                       tag: apriltag.Detection,
                       color: tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 1) -> None:
    
    corner_points = tag.corners.astype(int)
    center = tag.center.astype(int)

    cv2.polylines(image, [corner_points], isClosed = True, color = color, thickness = thickness)
    cv2.circle(image, center, 1, color, -1)

# draws +x and +y relative to the apriltag
def draw_axis(image: np.ndarray,
              tag: apriltag.Detection,
              color: tuple[int, int, int] = (0, 255, 0),
              arrow_scale: float = 0.1,
              text_offset: int = 10) -> None:

    # calculate the x and y vectors of the detected apriltag
    corner_points = tag.corners.astype(int)
    center = tag.center.astype(int)
    x_vec = corner_points[1] - corner_points[0]
    y_vec = corner_points[1] - corner_points[2]

    x_unit = x_vec / np.linalg.norm(x_vec)
    y_unit = y_vec / np.linalg.norm(y_vec)

    arrow_length = int(min(np.linalg.norm(x_vec), np.linalg.norm(y_vec)) * arrow_scale)

    x_label_pos = center + x_unit * arrow_length + np.array([text_offset, 0])
    y_label_pos = center + y_unit * arrow_length + np.array([0, text_offset])

    cv2.arrowedLine(image, tuple(center), tuple((center + x_unit * arrow_length).astype(int)), color, 2, tipLength = 0.1)
    cv2.arrowedLine(image, tuple(center), tuple((center + y_unit * arrow_length).astype(int)), color, 2, tipLength = 0.1)

    cv2.putText(image, "x", tuple(x_label_pos.astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
    cv2.putText(image, "y", tuple(y_label_pos.astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

# draws how far away the detected apriltag is
def draw_depth(image: np.ndarray,
               tag: apriltag.Detection,
               tag_size: float,
               camera_matrix: np.ndarray,
               distortion_coefficients: np.ndarray,
               color: tuple[int, int, int] = (0, 255, 0)) -> None:

    # define points for our object (apriltag) using the size of the apriltag
    objPoints = np.array([
        [-tag_size / 2, -tag_size / 2, 0],
        [ tag_size / 2, -tag_size / 2, 0],
        [ tag_size / 2,  tag_size / 2, 0],
        [-tag_size / 2,  tag_size / 2, 0]
    ])

    corner_points = tag.corners
    image_points = corner_points.reshape(-1, 2)

    # use PnP to find rotation and translation vectors, rvec and tvec, respectively
    res, rvec, tvec = cv2.solvePnP(objPoints, image_points, camera_matrix, distortion_coefficients)
    depth = tvec[2][0] # depth is found here

    if res and depth > 0:
        cv2.putText(image, f"Depth: {depth:.2f}m", tuple(corner_points[2].astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)