import numpy as np
from scipy import stats

# string to tuple
def color_to_rgb(color: str) -> tuple[int, int, int]:
    if color == "blue":
        return (255, 0, 0)

    if color == "green":
        return (0, 255, 0)

    if color == "red":
        return (0, 0, 255)

    return (0, 0, 0)

def calculate_confidence_window(data: np.ndarray,
                                confidence_interval_percent: float = 0.95,
                                two_tail: bool = True) -> float:

    tails = 2 if two_tail == True else 1

    std = np.std(data, axis = 0)

    # confidence interval
    z_score = stats.norm.ppf(q = 1 - (1 - confidence_interval_percent) / tails)
    confidence_window = z_score * std / np.sqrt(len(data))

    return confidence_window

def calculate_confidence(confidence_windows: np.ndarray,
                         k: float = 1) -> float:

    confidence = 1.0 - np.exp(-k / (np.mean(confidence_windows) + 1e-5))

    return confidence

# from a perspective matrix, H, get the angle, theta
def calculate_matrix_angle(H):

    # upper 2x2 normalized
    H = H / H[2, 2]
    A = H[0:2, 0:2]

    # SVD
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    theta = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi

    # in degrees
    return theta

# sort points in a more distinct way so its interpretable in 2D
# finds the center, and calculates the arctan angle of each point respective to the center
# returns the points, sorted
def sort_2D_points(points):
    angle_points = []
    center = np.mean(points, axis = 0)
    center_x, center_y = center.ravel()

    for point in points:
        x, y = point.ravel()
        dx = x - center_x
        dy = y - center_y
        theta = np.arctan2(dy, dx)
        angle_points.append([theta, x, y])

    # sort by theta so result will be ordered clockwise starting from quadrant 2
    angle_points.sort()
    sorted_points = np.array(angle_points)[:, 1:]

    return sorted_points

