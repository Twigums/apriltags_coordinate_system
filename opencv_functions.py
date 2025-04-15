# istg training a model wouldve been easier than doing this

import apriltag, cv2
import numpy as np
from utility_functions import sort_2D_points


# finds checkerboard image and uses cv2 to find camera calibration matrices and distortion coefficients
# from opencv camera calibration wiki
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

    # find robust corner points
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

# using an image, find edges and derives a convex hull to represent all edges
# convex hull is then represented as a polygon where corners can now be found
def find_corners(image,
                 ksize: tuple[int, int] = (5, 1),
                 number_corners: int = 4,
                 features_quality_level: float = 0.001,
                 thickness: int = 2) -> np.ndarray:

    height, width, _ = image.shape

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # use morphology to remove the thin lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    
    # invert so that lines are white so that we can get contours for them
    inverse_threshold = 255 - threshold
    
    # get external contours
    contours = cv2.findContours(inverse_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    # keep contours whose bounding boxes are greater than 1/4 in each dimension
    # draw them as white on black background
    good_contours = np.zeros((height, width), dtype = np.uint8)
    for contour in contours:
        contour_x, contour_y, rect_width, rect_height = cv2.boundingRect(contour)

        if rect_width > width / 4 and rect_height > height / 4:
            cv2.drawContours(good_contours, [contour], 0, 255, thickness)
            
    # get convex hull from contour image white pixels
    points = np.column_stack(np.where(good_contours.transpose() > 0))
    convex_hull_points = cv2.convexHull(points)
    
    # draw hull on copy of input and on black background
    convex_hull = np.zeros((height, width), dtype = np.uint8)
    cv2.drawContours(convex_hull, [convex_hull_points], 0, 255, thickness)
    
    # get corners from white hull points on black background
    min_dist = max(height, width) // 4
    corners = cv2.goodFeaturesToTrack(convex_hull, number_corners, features_quality_level, min_dist)

    return corners

# easily initialize sift + flann
def initialize_detector(flann_index_kdtree: int = 1,
                        flann_trees: int = 5,
                        flann_checks: int = 50) -> tuple[cv2.SIFT, cv2.FlannBasedMatcher]:

    sift = cv2.SIFT_create()
    
    index_params = dict(algorithm = flann_index_kdtree, trees = flann_trees)
    search_params = dict(checks = flann_checks)
     
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    return sift, flann

# uses sift and flann to match keypoints between template and target
# repeats "max_iterations" times until a good iou is found and returns a transformed ROI in the detected object's coords
# https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
def get_detection(sift: cv2.SIFT,
                  flann: cv2.FlannBasedMatcher,
                  template: np.ndarray, 
                  target: np.ndarray,
                  max_iterations: int = 10,
                  flann_k: int = 2,
                  min_match_count: int = 10,
                  matching_ratio_score: float = 0.7,
                  iou_threshold: float = 0.9,
                  ransac_threshold: float = 5.0,
                  return_metrics: bool = True) -> tuple[dict, dict]:
    
    # find the keypoints and descriptors with SIFT
    kp_template, des_template = sift.detectAndCompute(template, None)
    kp_target, des_target = sift.detectAndCompute(target, None)

    results = {"H": [],
               "transformed_box": [],
              }
    
    metrics = {"iou": [],
              }

    # i found that this code doesn't work for arbitrary x, y values
    # it seems much better if origin at (0, 0)
    h, w, _ = template.shape # not greyscale, so 3 channels
    box_points = np.float32([
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0]
        ]).reshape(-1, 1, 2)
    
    # exits when iou exceeds set threshold
    for i in range(max_iterations): 
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

            # apply transformation to rectangle
            transformed_box = cv2.perspectiveTransform(box_points, H)
            H_inv = np.linalg.inv(H)
            warped_target = cv2.warpPerspective(target, H_inv, (w, h), cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))

            # score prediction and return if good enough
            template_corners = find_corners(template)
            target_corners = find_corners(warped_target)

            sorted_template_corners = sort_2D_points(template_corners)
            sorted_target_corners = sort_2D_points(target_corners)

            iou = calculate_iou(sorted_template_corners, sorted_target_corners, (h, w))

            if return_metrics:
                metrics["iou"].append(iou)

            results["H"].append(H)
            results["transformed_box"].append(transformed_box)

            if iou >= iou_threshold:
                return results, metrics
         
        else:
            print(f"Not enough matches are found in iter {i}: {len(good_matches)}/{min_match_count}")
    
    return results, metrics

# iou calculation defined as intersection / union
# we can get this by drawing a polygon from the input points
def calculate_iou(points1: np.ndarray,
                  points2: np.ndarray,
                  dims: tuple[int, int]) -> float:

    points1 = points1.astype(np.int32)
    points2 = points2.astype(np.int32)

    mask1 = np.zeros((dims[0], dims[1]), dtype = np.uint8)
    mask2 = np.zeros((dims[0], dims[1]), dtype = np.uint8)
    
    cv2.fillPoly(mask1, [points1], 1)
    cv2.fillPoly(mask2, [points2], 1)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0

    return iou

# draws lines connecting to points on image
def draw_polygon(image: np.ndarray,
                 points: np.ndarray,
                 color: tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 3) -> None:

    cv2.polylines(image, [np.int32(points)], True, color = color, thickness = thickness)