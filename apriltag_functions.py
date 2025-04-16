import apriltag
import cv2
import numpy as np


# initialize apriltag detector with options
def apriltag_init() -> apriltag.Detector:
    options = apriltag.DetectorOptions(
        families = "tag36h11",
        border = 1,
        nthreads = 4,
        quad_decimate = 1.0,
        quad_blur = 0.0,
        refine_edges = True,
        refine_decode = False,
        refine_pose = False,
        debug = False,
        quad_contours = True)

    detector = apriltag.Detector()

    return detector

# try finding apriltags in the image
def detect_apriltag(detector: apriltag.Detector,
                    image: np.ndarray) -> list:

    result = detector.detect(image)

    if len(result) > 0:
        pass

    return result

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

    cv2.arrowedLine(image,
                    tuple(center),
                    tuple((center + x_unit * arrow_length).astype(int)),
                    color,
                    2,
                    tipLength = 0.1)
    cv2.arrowedLine(image,
                    tuple(center),
                    tuple((center + y_unit * arrow_length).astype(int)),
                    color,
                    2,
                    tipLength = 0.1)

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
    obj_points = np.array([
        [-tag_size / 2, -tag_size / 2, 0],
        [ tag_size / 2, -tag_size / 2, 0],
        [ tag_size / 2,  tag_size / 2, 0],
        [-tag_size / 2,  tag_size / 2, 0]
    ])

    corner_points = tag.corners
    image_points = corner_points.reshape(-1, 2)

    # use PnP to find rotation and translation vectors, rvec and tvec, respectively
    res, _, tvec = cv2.solvePnP(obj_points, image_points, camera_matrix, distortion_coefficients)
    depth = tvec[2][0] # depth is found here

    if res and depth > 0:
        cv2.putText(image,
                    f"Depth: {depth:.2f}m",
                    tuple(corner_points[2].astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    color,
                    2)
