import apriltag, cv2
import numpy as np


def get_camera_calibration(filepath):
    npzfile = np.load(filepath)
    camera_matrix = npzfile["camera_M"]
    distortion_coefficients = npzfile["distort_coeff"]

    return camera_matrix, distortion_coefficients

def apriltag_init():
    options = apriltag.DetectorOptions(
        families='tag36h11',
        border=1,
        nthreads=4,
        quad_decimate=1.0,
        quad_blur=0.0,
        refine_edges=True,
        refine_decode=False,
        refine_pose=False,
        debug=False,
        quad_contours=True)

    detector = apriltag.Detector()

    return detector

def color_to_rgb(color: str) -> tuple:
    if color == "blue":
        return (255, 0, 0)

    if color == "green":
        return (0, 255, 0)

    if color == "red":
        return (0, 0, 255)

    return ()

def img_to_np_gray(img: np.ndarray) -> np.ndarray:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    np_gray_img = np.asarray(gray_img)

    return np_gray_img


def detect_apriltag(detector, img: np.ndarray) -> list:
    res = detector.detect(img)

    if len(res) > 0:
        # print(res)
        pass

    return res

def draw_bbox(img: np.ndarray, tag, color) -> None:
    corner_points = tag.corners.astype(int)
    center = tag.center.astype(int)

    cv2.polylines(img, [corner_points], isClosed = True, color = color, thickness = 1)
    cv2.circle(img, center, 1, color, -1)

def draw_axis(img: np.ndarray, tag, arrow_scale: float, text_offset: int, color) -> None:
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

def draw_depth(img: np.ndarray, tag, tag_size: float, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray, color) -> None:
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

